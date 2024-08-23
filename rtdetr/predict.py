import gc
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

torchvision.disable_beta_transforms_warning()
import logging
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PyQt5.QtCore import Qt, QTimer
from torch import nn
from torch.cuda.amp import autocast
from torchvision.ops import boxes as box_ops
from tqdm import trange

from backbone.presnet import PResNet
from rtdetr import RTDETR
from rtdetr.hybrid_encoder import HybridEncoder
from rtdetr.rtdetr_decoder import RTDETRTransformer
from rtdetr.rtdetr_postprocessor import RTDETRPostProcessor


class fake_cargs:
    def __init__(self):
        self.interactive = True
        self.early_fusion = True
        self.late_fusion = False
        self.late_combine = False
        self.heatmap_style = False
        self.softmax = False
        self.norm_dist = False
        self.norm_batch = True
        self.no_iut = False
        self.attn_fusion = False
        self.attn_conv = True
        self.residual = True


class Predictor:

    @torch.no_grad()
    def __init__(self):
        logging.info("Initializing the Predictor.")
        self.model = None
        self.transform = T.Compose(
            [
                T.ToImageTensor(),
                T.ConvertDtype(),
            ]
        )
        self.cargs = fake_cargs()
        self.dummy = False
        self.post_processor = RTDETRPostProcessor(num_classes=2)

        self.current_img_size = (2400, 2400)

        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.cancellation_flag = False

    def set_cancellation_flag(self):
        self.cancellation_flag = True

    def check_cancellation(self):
        return self.cancellation_flag

    def load_model(self, checkpoint_path):
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        backbone = PResNet(18, variant="d", num_stages=4, return_idx=[1, 2, 3])
        backbone.eval()
        encoder = HybridEncoder(
            in_channels=[128, 256, 512],
            hidden_dim=256,
            expansion=0.5,
            num_encoder_layers=1,
            use_encoder_idx=[2],
            eval_spatial_size=[960, 960],
            feat_strides=[8, 16, 32],
            num_classes=2,
        )
        encoder.eval()
        decoder = RTDETRTransformer(
            eval_idx=-1,
            num_decoder_layers=3,
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            num_levels=3,
            eval_spatial_size=[960, 960],
            num_classes=2,
        )
        decoder.eval()
        model = RTDETR(backbone, encoder, decoder)

        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"])

        model.eval()
        model.cuda()

        self.model = model
        logging.info("Model loaded and set to evaluation mode.")

    def load_image(self, img_path):
        logging.info(f"Loading and processing image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        self.current_img_size = img.size
        img_torch = self.transform(img).unsqueeze(0).cuda()
        logging.debug(f"Transformed image shape: {img_torch.shape}")
        return img_torch

    def generate_samples(
        self, img_torch, bbox, cls_list, feature_index=[0, 1, 2], max_patch_size=2400
    ):
        logging.info("Generating samples from the image.")
        bbox_features = {"positive": [[], [], []], "negative": [[], [], []]}

        _, _, H, W = img_torch.shape
        print("Image Shape", img_torch.shape)

        # Calculate the number of patches needed
        num_patches_h = max(1, H // max_patch_size + (1 if H % max_patch_size else 0))
        num_patches_w = max(1, W // max_patch_size + (1 if W % max_patch_size else 0))

        patch_h = H // num_patches_h
        patch_w = W // num_patches_w

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y_start, y_end = i * patch_h, min((i + 1) * patch_h, H)
                x_start, x_end = j * patch_w, min((j + 1) * patch_w, W)

                patch = img_torch[:, :, y_start:y_end, x_start:x_end]

                with torch.no_grad(), torch.cuda.amp.autocast():
                    feature_maps = self.model.backbone(patch)

                for idx in feature_index:
                    feature_map = feature_maps[idx].squeeze(0)
                    C, FH, FW = feature_map.shape

                    for sample_box, cls in zip(bbox, cls_list):
                        # Convert bbox to patch coordinates
                        sample_box_patch = sample_box.clone()
                        sample_box_patch[0] -= x_start
                        sample_box_patch[1] -= y_start
                        sample_box_patch = torch.clamp(sample_box_patch, min=0)

                        # Skip if the box is not in this patch
                        if (
                            sample_box_patch[0] >= patch_w
                            or sample_box_patch[1] >= patch_h
                            or sample_box_patch[0] + sample_box_patch[2] <= 0
                            or sample_box_patch[1] + sample_box_patch[3] <= 0
                        ):
                            continue

                        sample_box_norm = (
                            sample_box_patch
                            / torch.tensor([patch_w, patch_h, patch_w, patch_h]).float()
                        )
                        sample_box_scaled = (
                            sample_box_norm * torch.tensor([FW, FH, FW, FH]).float()
                        )

                        cx, cy, w, h = sample_box_scaled

                        x1 = max(0, int(cx - w / 2))
                        x2 = min(FW, int(cx + w / 2))
                        y1 = max(0, int(cy - h / 2))
                        y2 = min(FH, int(cy + h / 2))

                        if x1 == x2:
                            x2 = min(FW, x2 + 1)
                        if y1 == y2:
                            y2 = min(FH, y2 + 1)

                        sample_2d = feature_map[:, y1:y2, x1:x2]

                        if sample_2d.size(1) > 5 or sample_2d.size(2) > 5:
                            sample_2d_max_pool = self.max_pool(
                                sample_2d.unsqueeze(0)
                            ).squeeze(0)
                            sample_2d = sample_2d_max_pool

                        if cls == 0:
                            bbox_features["negative"][idx].append(sample_2d)
                        else:
                            bbox_features["positive"][idx].append(sample_2d)

                # Clear unnecessary tensors
                del feature_maps
                torch.cuda.empty_cache()

        return bbox_features

    def split_image(
        self, img_torch: torch.Tensor, patch_size: int = 960, overlap: int = 160
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        logging.info("Splitting image into overlapping patches.")
        _, _, H, W = img_torch.shape
        stride = patch_size - overlap

        parts = []
        locations = []

        for y in range(0, H - overlap, stride):
            for x in range(0, W - overlap, stride):
                end_y = min(y + patch_size, H)
                end_x = min(x + patch_size, W)
                part = img_torch[:, :, y:end_y, x:end_x]

                if part.size(2) < patch_size or part.size(3) < patch_size:
                    padded_part = torch.zeros(
                        (1, 3, patch_size, patch_size), device=part.device
                    )
                    padded_part[:, :, : part.size(2), : part.size(3)] = part
                    part = padded_part

                parts.append(part)
                locations.append((x, y))

        return parts, locations

    def nms(self, boxes, scores, iou_threshold, normalize=True):
        logging.info("Applying NMS.")
        # Convert boxes to (x, y, w, h) format
        if normalize:
            # norm using current image size
            boxes = boxes / torch.tensor(
                [
                    self.current_img_size[0],
                    self.current_img_size[1],
                    self.current_img_size[0],
                    self.current_img_size[1],
                ],
                device=boxes.device,
            )
        boxes = box_ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        # Apply Non-Maximum Suppression
        keep = box_ops.nms(boxes, scores, iou_threshold)
        return keep

    def convert_input_boxes(self, input_boxes):
        cls_list = []
        bbox = []

        for box in input_boxes:
            if box["type"] == "box":
                x, y, width, height, cls = (
                    box["x"],
                    box["y"],
                    box["width"],
                    box["height"],
                    box.get("class", 0),
                )
            elif box["type"] == "point":
                x, y, cls = (
                    box["x"],
                    box["y"],
                    box.get("class", 0),
                )
                width = height = 34
            else:
                raise ValueError("Invalid box type")

            cls_list.append(cls)
            bbox.append([x, y, width, height])

        return bbox, cls_list

    @torch.no_grad()
    def predict(self, image_path, input_boxes, progress_callback=None):
        logging.info(f"Starting prediction for {image_path}")
        logging.info(f"Bounding box (Input): {input_boxes}")

        bbox, cls_list = self.convert_input_boxes(input_boxes)

        if self.dummy:
            logging.info("Dummy mode enabled. Skipping prediction.")
            return [], []

        patch_size = 960
        overlap = 160
        iou_threshold = 0.3
        batch_size = 1  # Adjust this based on your GPU memory

        bbox = torch.tensor(bbox)

        # Load and split the image into parts
        img_torch = self.load_image(image_path)
        samples = self.generate_samples(img_torch, bbox, cls_list)
        parts, locations = self.split_image(img_torch, patch_size, overlap)

        # Process attention masks
        _, _, H, W = img_torch.shape
        total_attn_mask = torch.zeros((H, W), device=img_torch.device)
        total_attn_mask_norm = torch.zeros((H, W), device=img_torch.device)

        # Process parts in batches
        all_results = []
        for i in trange(0, len(parts), batch_size):
            batch_parts = parts[i : i + batch_size]
            batch_locations = locations[i : i + batch_size]

            stacked_parts = torch.cat(batch_parts, dim=0).cuda()
            logging.info(
                f"Processing batch {i//batch_size + 1}, shape: {stacked_parts.shape}"
            )

            B = stacked_parts.size(0)
            sel_cls = [1] * B

            self.model.eval()

            outputs, attn_masks = self.model.forward_inference(
                stacked_parts,
                [samples] * B,
                cargs=self.cargs,
                selected_class=sel_cls,
            )

            # for ii, feat in enumerate(attn_masks):
            #     h = w = int(math.sqrt(feat.shape[1]))
            #     attn_masks_resized = feat.permute(0, 2, 1).reshape(-1, h, w)
            #     attn_masks_resized = F.interpolate(
            #         attn_masks_resized.unsqueeze(1),
            #         size=(patch_size, patch_size),
            #         mode="bilinear",
            #         align_corners=False,
            #     ).squeeze(1)

            #     for idx, (x, y) in enumerate(batch_locations):
            #         attn_mask = attn_masks_resized[idx]
            #         end_y = min(y + patch_size, H)
            #         end_x = min(x + patch_size, W)
            #         mask_h, mask_w = end_y - y, end_x - x

            #         plt.figure()
            #         plt.imshow(attn_mask.cpu().numpy())
            #         plt.axis("off")
            #         plt.tight_layout()
            #         plt.savefig(f"attention_maps/attn_{ii}_{idx}.png")
            #         plt.close()

            #         total_attn_mask[y:end_y, x:end_x] += attn_mask[:mask_h, :mask_w]
            #         total_attn_mask_norm[y:end_y, x:end_x] += 1

            # total_attn_mask_norm[total_attn_mask_norm == 0] = 1
            # total_attn_mask /= total_attn_mask_norm

            selected_classes = torch.tensor(
                [cls == "Crater" for cls in cls_list]
            ).cuda()
            selected_classes = selected_classes.repeat(B)
            orig_target_sizes = (
                torch.tensor([patch_size, patch_size]).unsqueeze(0).repeat(B, 1).cuda()
            )

            results = self.post_processor(
                outputs, orig_target_sizes, selected_classes, loss=True
            )

            # move results to CPU
            for ii in range(len(results)):
                results[ii]["boxes"] = results[ii]["boxes"].detach().cpu()
                results[ii]["scores"] = results[ii]["scores"].detach().cpu()
                results[ii]["labels"] = results[ii]["labels"].detach().cpu()

            all_results.extend(results)

            # Clear unnecessary tensors
            del stacked_parts, outputs, attn_masks
            torch.cuda.empty_cache()
            gc.collect()

            if progress_callback:
                progress = (i) / len(parts)
                progress_callback(progress)

            # Check for cancellation (you'll need to implement this mechanism)
            if self.check_cancellation():
                return None, None, None  # or handle cancellation appropriately

        # # Save attention mask visualization
        # plt.figure()
        # plt.imshow(total_attn_mask.cpu().numpy())
        # plt.axis("off")
        # plt.tight_layout()
        # plt.savefig("attention_maps/attn_combined.png")
        # plt.close()

        # Process prediction results
        selected_classes = torch.tensor([cls == "Crater" for cls in cls_list]).cuda()
        selected_classes = selected_classes.repeat(B)
        orig_target_sizes = (
            torch.tensor([patch_size, patch_size]).unsqueeze(0).repeat(B, 1).cuda()
        )

        # results = self.post_processor(
        #     outputs, orig_target_sizes, selected_classes, loss=True
        # )

        combined_boxes = []
        combined_classes = []
        combined_scores = []

        for idx, (x, y) in enumerate(locations):
            scores = all_results[idx]["scores"]
            boxes = all_results[idx]["boxes"]
            labels = all_results[idx]["labels"]

            selected = scores > 0.1

            selected_boxes = boxes[selected]
            selected_classes = labels[selected]
            selected_scores = scores[selected]

            boxes = box_ops.box_convert(selected_boxes, in_fmt="xyxy", out_fmt="cxcywh")
            boxes[:, 0] += x
            boxes[:, 1] += y

            combined_boxes.append(boxes)
            combined_classes.append(selected_classes)
            combined_scores.append(selected_scores)

        selected_boxes = torch.cat(combined_boxes, dim=0)
        selected_classes = torch.cat(combined_classes)
        selected_scores = torch.cat(combined_scores)

        # Apply NMS
        keep = self.nms(selected_boxes, selected_scores, iou_threshold)

        # Filter out non-maximal bounding boxes
        selected_boxes = selected_boxes[keep]
        selected_scores = selected_scores[keep]

        # Normalize boxes to image dimensions
        selected_boxes = selected_boxes / torch.tensor(
            [W, H, W, H], device=selected_boxes.device
        )

        selected_boxes = selected_boxes.tolist()
        selected_scores = selected_scores.tolist()
        selected_classes = selected_classes[keep].tolist()

        print(f"Number of results: {len(selected_boxes)}")
        len_high_scores = len([score for score in selected_scores if score > 0.8])
        print(f"Number of high scores: {len_high_scores}")
        len_medium_scores = len([score for score in selected_scores if score > 0.5])
        print(f"Number of medium scores: {len_medium_scores}")

        return selected_boxes, selected_scores, selected_classes


# # if __name__ == "__main__":
# checkpoint_path = "/caa/Homes01/mburges/WACV_IA_Experiments/RT-DETR/rtdetr_pytorch/itacv_output/rtdetr_r18vd_6x_chai_interactive_super_late_fusion__new_mask_style/checkpoint0044.pth"

# predictor = Predictor()
# predictor.load_model(checkpoint_path)

# bbox = torch.tensor([350.0, 2321.0, 34.0, 34.0])
# bbox /= 2400

# # Test the predict function
# results = predictor.predict("static/images/1943_09_24_Ludwigshafen2017_60.png", bbox)
# print(f"Number of results: {len(results)}")
# print(results)
# print("Predicting on image parts...")
