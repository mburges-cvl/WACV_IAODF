"""by lyuwenyu
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

__all__ = [
    "RTDETR",
]

avg_box_sizes = {
    "CHAI": torch.tensor([[0.0480, 0.0480]]),
    "TINY_DOTA": torch.tensor(
        [
            [0.09083341807126999, 0.089743971824646],
            [0.05295756459236145, 0.04920492321252823],
            [0.01879558525979519, 0.018404260277748108],
            [0.04812614992260933, 0.04572305455803871],
            [0.039661210030317307, 0.04022033140063286],
            [0.030605386942625046, 0.02920844778418541],
            [0.045160744339227676, 0.04321549832820892],
            [0.04117812216281891, 0.07875379920005798],
        ]
    ),
    "AITOD": torch.tensor(
        [
            [0.03596508875489235, 0.034514445811510086],
            [0.021389158442616463, 0.021909179165959358],
            [0.01936325803399086, 0.019098499789834023],
            [0.01883494108915329, 0.018373863771557808],
            [0.024061433970928192, 0.021983789280056953],
            [0.017603645101189613, 0.01579034887254238],
            [0.010148308239877224, 0.020165830850601196],
            [0.013948863372206688, 0.018437499180436134],
        ]
    ),
    # "SARDET": torch.tensor(
    #     [
    #         [0.04, 0.04],
    #         [0.05, 0.05],
    #         [0.025, 0.023],
    #         [0.035, 0.03],
    #         [0.05, 0.05],
    #         [0.04, 0.04],
    #     ]
    # ),
    "SARDET": torch.tensor(
        [
            [0.08148498088121414, 0.07593412697315216],
            [0.05516878515481949, 0.049625564366579056],
            [0.024230631068348885, 0.02385377697646618],
            [0.03486877307295799, 0.029995020478963852],
            [0.09196650236845016, 0.0965944454073906],
            [0.043318383395671844, 0.04195518419146538],
        ]
    ),
}


class RTDETR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder,
        decoder,
        multi_scale=None,
        pyramid_point=True,
        skip_removal=False,
        dataset_type=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.user_input = False
        self.pyramid_point = pyramid_point
        self.skip_removal = skip_removal

        if dataset_type is not None:
            self.avg_box_sizes = avg_box_sizes[dataset_type]

    def forward_inference(
        self,
        input_image,
        bbox_feats,
        selected_class,
        cargs=None,
    ):
        backbone_features = self.backbone(input_image)

        if cargs.interactive:
            # labels = torch.ones(
            #     len(bbox_feats), len(bbox_feats[0][1]), device=input_image.device
            # )

            # for i_batch, batch_feats in enumerate(bbox_feats):
            #     print("Batch Feats: ", len(batch_feats["positive"]))
            #     for idx, scale_feats in enumerate(batch_feats["positive"]):
            #         for idx, bbox_feats in enumerate(scale_feats):
            #             for idx, bbox_feat in enumerate(bbox_feats):
            #                 print("BoxShape", bbox_feat.shape)

            output_features, attn_weights_enc, bbox_features, proj_feats = self.encoder(
                backbone_features,
                cargs=cargs,
                batch_bbox_feats=bbox_feats,
                selected_class=selected_class,
                # labels=labels,
            )

            decoded_features = self.decoder(
                output_features,
                cargs=cargs,
                attn_weights=attn_weights_enc,
            )

        else:
            masks = None
            selected_bboxes = None
            attn_weights_enc = None
            bbox_features = None
            proj_feats = None

            output_features, _, _, _ = self.encoder(backbone_features, None, cargs)

            decoded_features = self.decoder(
                output_features,
                cargs,
            )

        return decoded_features, attn_weights_enc

    def forward(
        self,
        input_image,
        targets=None,
        negative_targets={},
        cargs=None,
        max_boxes_per_class=3,
        eval=False,
        eval_boxes=None,
        viz_attn=False,
        epoch=0,
        prev_missed_preds=None,
    ):
        all_masks_bbox = None
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            input_image = F.interpolate(input_image, size=[sz, sz])

            assert input_image.size(-1) == input_image.size(-2), input_image.size()

        backbone_features = self.backbone(input_image)

        # feature_maps = [
        #     feats[0].detach().cpu().numpy().mean(0) for feats in backbone_features
        # ]

        # # Setup the 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # feature_maps.sort(key=lambda x: x.shape[0], reverse=True)

        # # Plot each feature map
        # for i, fm in enumerate(feature_maps):
        #     x, y = np.meshgrid(range(fm.shape[0]), range(fm.shape[1]))
        #     x = x - fm.shape[0] // 2  # Center the feature map in the x-direction
        #     y = y - fm.shape[1] // 2  # Center the feature map in the y-direction
        #     z = np.full(fm.shape, i / 2)
        #     ax.plot_surface(
        #         x,
        #         y,
        #         z,
        #         facecolors=plt.cm.viridis(fm / fm.max()),
        #         rstride=1,
        #         cstride=1,
        #         shade=False,
        #     )
        # # ticks
        # # ax.
        # # ax.set_xlabel("X")
        # # ax.set_ylabel("Y")
        # # ax.set_zlabel("Scale")
        # # ax.set_zlim(0, 3)
        # ax.axis("off")
        # ax.view_init(elev=15, azim=0, roll=90)
        # plt.tight_layout()
        # plt.savefig(
        #     f"plots/feature_maps_{epoch}.png",
        #     bbox_inches="tight",
        #     dpi=300,
        #     transparent=True,
        # )

        # exit()

        if cargs.interactive:
            (
                all_masks_bbox,
                selected_masks_bbox,
                bbox_features,
                selected_bboxes,
                selected_class,
            ) = self.handle_interactive_mode(
                backbone_features,
                targets,
                max_boxes_per_class,
                eval_boxes,
                input_image,
                cargs=cargs,
                negative_targets=negative_targets,
                prev_missed_preds=prev_missed_preds,
            )

            # bbox_features --> List per Batch -> Dict for positive and negative ->
            # List of scales (3 per resnet) -> List of bbox features

            if cargs.comb_ex:
                combined_feats = {
                    "positive": [[] for _ in range(len(backbone_features))],
                    "negative": [[] for _ in range(len(backbone_features))],
                }

                for i_batch, batch_feats in enumerate(bbox_features):
                    for idx, scale_feats in enumerate(batch_feats["positive"]):
                        combined_feats["positive"][idx].extend(scale_feats)

                    for idx, scale_feats in enumerate(batch_feats["negative"]):
                        combined_feats["negative"][idx].extend(scale_feats)

                bbox_features = [combined_feats for _ in range(len(bbox_features))]

            encoded_features, attn_weights_enc, encoded_bbox_features, proj_feats = (
                self.encoder(
                    backbone_features,
                    bbox_features,
                    cargs=cargs,
                    viz_attn=viz_attn,
                    selected_class=selected_class,
                )
            )

            if not cargs.all_cls:
                # filter out all other classes
                for i, (target, relevant_class) in enumerate(
                    zip(targets, selected_class)
                ):
                    if relevant_class is None:
                        target["labels"] = target["labels"].new_empty(
                            (0,), dtype=torch.long
                        )
                        target["boxes"] = target["boxes"].new_empty((0, 4))
                    else:
                        mask = target["labels"] == relevant_class
                        target["labels"] = torch.ones_like(target["labels"][mask])
                        target["boxes"] = target["boxes"][mask]

            # filter out all selected bboxes
            # if not cargs.sim_point and not cargs.sim_dyn_point:
            #     print("filtering")

            #     for i, (target, sel_bboxes) in enumerate(zip(targets, selected_bboxes)):
            #         if sel_bboxes["boxes"].numel() == 0:
            #             continue

            #         mask = torch.all(
            #             target["boxes"].unsqueeze(1)
            #             == sel_bboxes["boxes"].unsqueeze(0),
            #             dim=-1,
            #         ).any(dim=1)
            #         target["labels"] = target["labels"][~mask]
            #         target["boxes"] = target["boxes"][~mask]

            decoded_features = self.decoder(
                encoded_features,
                targets,
                cargs,
                attn_weights_enc,
            )
        else:
            selected_bboxes = None
            attn_weights_enc = None
            selected_class = None
            proj_feats = None
            selected_masks_bbox = None

            encoded_features, _, _, _ = self.encoder(backbone_features, None, cargs)

            decoded_features = self.decoder(
                encoded_features,
                targets,
                cargs,
            )

        return (
            decoded_features,
            selected_bboxes,
            encoded_features,
            all_masks_bbox,
            attn_weights_enc,
            selected_class,
            proj_feats,
            selected_masks_bbox,
        )

    def handle_interactive_mode(
        self,
        backbone_features,
        targets,
        max_boxes_per_class,
        eval_boxes=None,
        input_image=None,
        cargs=None,
        negative_targets={},
        prev_missed_preds=None,
    ):

        B, C, H, W = input_image.shape

        # print("Max Boxes Per Class: ", max_boxes_per_class)

        all_mask_bboxes = torch.zeros((B, 2, H, W), device=backbone_features[0].device)
        selected_mask_bboxes = torch.zeros(
            (B, 2, H, W), device=backbone_features[0].device
        )
        selected_class = [None] * B
        selected_bboxes = [
            {
                "boxes": torch.zeros([0, 4], device=backbone_features[0].device),
                "labels": torch.zeros([0], device=backbone_features[0].device),
            }
            for _ in range(B)
        ]
        batch_encoded_features = [{"positive": [], "negative": []} for _ in range(B)]

        for i_batch in range(B):
            image_id = targets[i_batch]["image_id"].item()
            available_classes, counts = torch.unique(
                targets[i_batch]["labels"], return_counts=True
            )

            if len(available_classes) == 0:
                continue

            random_class = self.select_random_class(
                available_classes, counts, eval_boxes, cargs
            )
            selected_class[i_batch] = random_class

            class_mask = targets[i_batch]["labels"] == random_class
            image_bboxes = targets[i_batch]["boxes"][class_mask]
            image_labels = targets[i_batch]["labels"][class_mask]

            all_negative_boxes = self.get_negative_boxes(
                negative_targets,
                image_id,
                image_bboxes,
                targets[i_batch],
                class_mask,
                eval_boxes,
                max_boxes_per_class,
            )

            num_bboxes = self.calculate_num_bboxes(
                len(image_bboxes), max_boxes_per_class, eval_boxes, cargs
            )
            selected_indices = self.select_indices(
                num_bboxes, len(image_bboxes), eval_boxes, cargs
            )

            bboxes_sel, labels_sel = (
                image_bboxes[selected_indices],
                image_labels[selected_indices],
            )

            self.update_masks(
                all_mask_bboxes,
                selected_mask_bboxes,
                all_negative_boxes,
                i_batch,
                image_bboxes,
                bboxes_sel,
                W,
                H,
                random_class,
                cargs,
            )

            image_encoded_features_positive = self.encode_positive_features(
                backbone_features, bboxes_sel, i_batch, selected_class[i_batch], cargs
            )

            image_encoded_features_negative = self.encode_negative_features(
                backbone_features, all_negative_boxes, i_batch, cargs
            )

            # print(
            #     num_bboxes,
            #     ":",
            #     len(image_encoded_features_positive[0]),
            #     len(image_encoded_features_positive[1]),
            #     len(image_encoded_features_positive[2]),
            #     "|",
            #     len(image_encoded_features_negative[0]),
            #     len(image_encoded_features_negative[1]),
            #     len(image_encoded_features_negative[2]),
            # )

            selected_bboxes[i_batch] = {"boxes": bboxes_sel, "labels": labels_sel}
            batch_encoded_features[i_batch] = {
                "positive": image_encoded_features_positive,
                "negative": image_encoded_features_negative,
            }

        return (
            all_mask_bboxes,
            selected_mask_bboxes,
            batch_encoded_features,
            selected_bboxes,
            selected_class,
        )

    def select_random_class(self, available_classes, counts, eval_boxes, cargs):
        if eval_boxes is not None:
            return torch.tensor(cargs.eval_id, device=counts.device)
        else:
            return random.choice(available_classes)

    def get_negative_boxes(
        self,
        negative_targets,
        image_id,
        image_bboxes,
        target,
        class_mask,
        eval_boxes,
        max_boxes_per_class,
    ):

        if eval_boxes is None:
            eval_boxes = random.randint(0, max_boxes_per_class)

        all_negative_boxes = []
        if image_id in negative_targets:
            negative_boxes, negative_scores, negative_labels = (
                negative_targets[image_id]["boxes"],
                negative_targets[image_id]["scores"],
                negative_targets[image_id]["labels"],
            )

            if len(image_bboxes) != 0:

                iou, _ = box_iou(
                    box_cxcywh_to_xyxy(negative_boxes),
                    box_cxcywh_to_xyxy(image_bboxes),
                )

                overlaps = ~(iou.max(dim=1).values > 0)

                bad_boxes = negative_boxes[overlaps]

                sorted_scores, indices = torch.sort(
                    negative_scores[overlaps], descending=True
                )

            else:
                bad_boxes = negative_boxes

                sorted_scores, indices = torch.sort(negative_scores, descending=True)

            sorted_boxes = bad_boxes[indices][:eval_boxes]
            sorted_scores = sorted_scores[:eval_boxes]

            high_score = sorted_scores > 0.3
            sorted_boxes = sorted_boxes[high_score]
            all_negative_boxes.append(sorted_boxes)

        else:
            bad_boxes = target["boxes"][~class_mask][:eval_boxes]
            all_negative_boxes.append(bad_boxes)

        return (
            torch.cat(all_negative_boxes, dim=0)
            if len(all_negative_boxes) > 0
            else torch.tensor([])
        )

    def calculate_num_bboxes(self, num_bboxes, max_boxes_per_class, eval_boxes, cargs):
        if eval_boxes is not None:
            return eval_boxes
        return random.randint(1, max_boxes_per_class)

    def select_indices(self, num_bboxes, num_image_bboxes, eval_boxes, cargs):
        num_bboxes = min(num_bboxes, num_image_bboxes)
        if eval_boxes is not None and not cargs.random_boxes:
            return torch.arange(num_bboxes)
        else:
            return torch.randperm(num_image_bboxes)[:num_bboxes]

    def update_masks(
        self,
        all_mask_bboxes,
        selected_mask_bboxes,
        all_negative_boxes,
        i_batch,
        image_bboxes,
        bboxes_sel,
        W,
        H,
        random_class,
        cargs,
    ):
        bboxes_xyxy = self.cxcywh_to_xyxy(image_bboxes)
        bboxes_xyxy[:, [0, 2]] *= W
        bboxes_xyxy[:, [1, 3]] *= H

        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = bbox.int()
            all_mask_bboxes[i_batch, 1, y1:y2, x1:x2] = 1

        if len(bboxes_sel) > 0:
            bboxes_xyxy = self.cxcywh_to_xyxy(bboxes_sel)
            bboxes_xyxy[:, [0, 2]] *= W
            bboxes_xyxy[:, [1, 3]] *= H

            for bbox, label in zip(bboxes_xyxy, bboxes_sel):
                x1, y1, x2, y2 = bbox.int()
                if cargs.fancy_plots:
                    x1, y1, x2, y2 = self.adjust_bbox(
                        random_class, x1, y1, x2, y2, W, H
                    )
                selected_mask_bboxes[i_batch, 1, y1:y2, x1:x2] = 1

        if len(all_negative_boxes) > 0:
            negative_bboxes = all_negative_boxes
            negative_bboxes_xyxy = self.cxcywh_to_xyxy(negative_bboxes)
            negative_bboxes_xyxy[:, [0, 2]] *= W
            negative_bboxes_xyxy[:, [1, 3]] *= H

            for bbox in negative_bboxes_xyxy:
                x1, y1, x2, y2 = bbox.int()
                selected_mask_bboxes[i_batch, 0, y1:y2, x1:x2] = 1
                all_mask_bboxes[i_batch, 0, y1:y2, x1:x2] = 1

    def adjust_bbox(self, random_class, x1, y1, x2, y2, W, H):
        size = self.avg_box_sizes[random_class - 1]
        w, h = size
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = w * W, h * H
        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

    def encode_positive_features(
        self, backbone_features, bboxes_sel, i_batch, selected_class, cargs
    ):
        return [
            [
                self.crop_image(
                    scale[i_batch],
                    bbox,
                    class_id=selected_class,
                    cargs=cargs,
                    idx=idx,
                    total_scales=len(backbone_features),
                )
                for bbox in bboxes_sel
            ]
            for idx, scale in enumerate(backbone_features)
        ]

    def encode_negative_features(
        self, backbone_features, all_negative_boxes, i_batch, cargs
    ):
        return [
            [
                self.crop_image(
                    scale[i_batch],
                    bbox,
                    class_id=None,
                    cargs=cargs,
                    idx=idx,
                    total_scales=len(backbone_features),
                )
                for bbox in all_negative_boxes
            ]
            for idx, scale in enumerate(backbone_features)
        ]

    def crop_image(
        self, input_image, bbox, class_id=None, cargs=None, idx=None, total_scales=None
    ):
        ic, ih, iw = input_image.shape

        bbox_scaled = bbox * torch.tensor(
            [iw, ih, iw, ih], device=bbox.device, dtype=torch.float
        )

        # print(bbox_scaled)

        x, y, w, h = bbox_scaled

        if cargs.sim_dyn_point is True and class_id is not None:
            assert self.avg_box_sizes is not None

            size = self.avg_box_sizes[class_id - 1]

            w, h = size * torch.tensor([iw, ih], device=size.device, dtype=torch.float)

        elif cargs.sim_dyn_point is True and class_id is None:
            assert self.avg_box_sizes is not None

            size = self.avg_box_sizes[0]

            w, h = size * torch.tensor([iw, ih], device=size.device, dtype=torch.float)

        start_x, end_x = int(x - (w / 2)), int(x + (w / 2))
        start_y, end_y = int(y - (h / 2)), int(y + (h / 2))

        start_x = max(0, start_x)
        start_y = max(0, start_y)

        if start_x == end_x:
            end_x += 1

        if start_y == end_y:
            end_y += 1

        end_x = min(iw, end_x)
        end_y = min(ih, end_y)

        if start_x == end_x:
            start_x -= 1

        if start_y == end_y:
            start_y -= 1

        # print(start_x, start_y, end_x, end_y)

        # print(input_image.shape)

        cropped_img = input_image[
            :,
            int(start_y) : int(end_y),
            int(start_x) : int(end_x),
        ]

        if cargs.sim_point:
            C, H, W = cropped_img.shape

            center_H, center_W = H // 2, W // 2

            if self.pyramid_point is True:

                # Define the neighborhood size based on idx
                if idx is None:
                    neighborhood_size = 1
                else:
                    if total_scales == 2:
                        neighborhood_size = (2 - idx) * 2 + 1
                    elif total_scales == 3:
                        neighborhood_size = (2 - idx) * 2 + 1
                    elif total_scales == 4:
                        neighborhood_size = (3 - idx) * 2 + 1
                    else:
                        raise ValueError("Invalid number of scales")

                # Calculate the start and end indices for the neighborhood
                start_H = max(center_H - (neighborhood_size // 2), 0)
                end_H = min(center_H + (neighborhood_size // 2) + 1, H)
                start_W = max(center_W - (neighborhood_size // 2), 0)
                end_W = min(center_W + (neighborhood_size // 2) + 1, W)

                # Extract the neighborhood
                cropped_img = cropped_img[:, start_H:end_H, start_W:end_W]
            else:
                cropped_img = cropped_img[:, center_H, center_W]
                cropped_img = cropped_img.unsqueeze(-1).unsqueeze(-1)

        # print(cropped_img.shape)

        # cropped_img = torch.rand_like(cropped_img)

        assert cropped_img.size(-2) > 0 and cropped_img.size(-1) > 0, (
            cropped_img.shape,
            bbox_scaled,
            bbox,
            start_x,
            start_y,
            end_x,
            end_y,
        )

        return cropped_img

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    def cxcywh_to_xyxy(self, bboxes):
        # Convert from cxcywh to xyxy
        bboxes_xyxy = torch.zeros_like(bboxes)
        bboxes_xyxy[:, 0] = bboxes[:, 0] - 0.5 * bboxes[:, 2]  # top-left x
        bboxes_xyxy[:, 1] = bboxes[:, 1] - 0.5 * bboxes[:, 3]  # top-left y
        bboxes_xyxy[:, 2] = bboxes[:, 0] + 0.5 * bboxes[:, 2]  # bottom-right x
        bboxes_xyxy[:, 3] = bboxes[:, 1] + 0.5 * bboxes[:, 3]  # bottom-right y

        return bboxes_xyxy
