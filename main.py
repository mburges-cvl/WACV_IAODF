import json
import os
import sys

from PIL import Image
from PyQt5 import sip

Image.MAX_IMAGE_PIXELS = None

from PyQt5.QtCore import (
    QObject,
    QPointF,
    QRectF,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressDialog,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from rtdetr.predict import Predictor


class ProgressManager(QObject):
    update_progress = pyqtSignal(int, int, str)

    def __init__(self):
        super().__init__()
        self.progress_dialog = None
        self.update_progress.connect(self.update_progress_dialog)

    def create_progress_dialog(self, min_value, max_value, label_text):
        self.progress_dialog = QProgressDialog(
            label_text, "Cancel", min_value, max_value
        )
        self.progress_dialog.setWindowTitle("Please Wait")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setFixedSize(300, 100)
        self.progress_dialog.show()

    @pyqtSlot(int, int, str)
    def update_progress_dialog(self, value, max_value, label_text):
        if self.progress_dialog:
            self.progress_dialog.setRange(0, max_value)
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(label_text)
        else:
            print("Warning: Progress dialog not created")  # Debug print

    def close_progress_dialog(self):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None


class CustomGraphicsRectItem(QGraphicsRectItem):
    def __init__(self, rect, box_data, parent=None):
        super().__init__(rect, parent)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.box_data = box_data


class PredictionWorker(QObject):
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(float)

    def __init__(self, predictor, image_path, bboxes):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path
        self.bboxes = bboxes

    def run(self):
        result = self.predictor.predict(
            self.image_path, self.bboxes, self.progress.emit
        )
        if result is not None:
            selected_boxes, selected_scores, selected_classes = result
            self.finished.emit(selected_boxes, selected_scores, selected_classes)
        else:
            self.finished.emit(None, None, None)


class CustomGraphicsView(QGraphicsView):
    boxDrawn = pyqtSignal(QRectF)
    pointDrawn = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.panning = False
        self.lastPoint = QPointF()
        self.currentRect = None
        self.drawMode = "box"
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.updateCursor()

    def updateCursor(self):
        if self.drawMode == "box":
            self.setCursor(Qt.CrossCursor)
        elif self.drawMode == "point":
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def setDrawMode(self, mode):
        self.drawMode = mode
        self.updateCursor()

    def enterEvent(self, event):
        self.updateCursor()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
        else:
            factor = 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsProxyWidget):
            # If it's a button (proxy widget), let the default handler take care of it
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = self.mapToScene(event.pos())
            if self.drawMode == "box":
                self.currentRect = QGraphicsRectItem(
                    QRectF(self.lastPoint, self.lastPoint)
                )
                self.scene().addItem(self.currentRect)
        elif event.button() == Qt.MiddleButton:
            self.panning = True
            self.lastPoint = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.drawMode == "box":
            newPoint = self.mapToScene(event.pos())
            rect = QRectF(self.lastPoint, newPoint).normalized()
            self.currentRect.setRect(rect)
        elif self.panning:
            delta = self.mapToScene(event.pos()) - self.mapToScene(self.lastPoint)
            self.lastPoint = event.pos()
            self.translate(delta.x(), delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsProxyWidget):
            # If it's a button (proxy widget), let the default handler take care of it
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.drawMode == "box":
                finalRect = self.currentRect.rect()
                self.boxDrawn.emit(finalRect)
                self.scene().removeItem(self.currentRect)
                self.currentRect = None
            elif self.drawMode == "point":
                point = self.mapToScene(event.pos())
                self.pointDrawn.emit(point)
        elif event.button() == Qt.MiddleButton:
            self.panning = False
            self.updateCursor()
        super().mouseReleaseEvent(event)

    def drawPoint(self, point):
        # Outer white circle
        outer_circle = QGraphicsEllipseItem(point.x() - 5, point.y() - 5, 10, 10)
        outer_circle.setPen(QPen(QColor(Qt.white), 2))
        outer_circle.setBrush(QBrush(Qt.transparent))
        self.scene().addItem(outer_circle)

        # Inner yellow circle
        inner_circle = QGraphicsEllipseItem(point.x() - 3, point.y() - 3, 6, 6)
        inner_circle.setPen(QPen(Qt.NoPen))
        inner_circle.setBrush(QBrush(QColor(Qt.yellow)))
        self.scene().addItem(inner_circle)

        return [outer_circle, inner_circle]

    def setScene(self, scene):
        super().setScene(scene)
        self.updateSceneRect()

    def updateSceneRect(self):
        if self.scene():
            image_rect = self.scene().itemsBoundingRect()
            margin = max(image_rect.width(), image_rect.height())
            new_rect = image_rect.adjusted(-margin, -margin, margin, margin)
            self.setSceneRect(new_rect)


class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.image_directory = None
        self.bounding_boxes = {}
        self.predicted_boxes = {}
        self.undoStack = []
        self.redoStack = []
        self.annotation_items = []
        self.cls_dict = {0: "not-Crater", 1: "Crater"}
        self.current_class = 1
        self.class_buttons = {}

        self.initUI()

        predictor = Predictor()
        checkpoint_path = "weights\checkpoint_chai.pth"
        predictor.load_model(checkpoint_path)
        self.prediction_worker = None

        self.predictor = predictor

        self.progress_manager = ProgressManager()
        # self.progress_manager.update_progress.connect(
        #     self.progress_manager.update_progress_dialog
        # )
        self.prediction_thread = QThread()

        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.update_filtered_predictions)

    def initUI(self):
        self.setWindowTitle("Comprehensive PyQt Image Annotator")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Left panel for image list and controls
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        load_button = QPushButton("Select image folder")
        load_button.clicked.connect(self.load_images)
        left_layout.addWidget(load_button)

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image)
        left_layout.addWidget(self.image_list)

        # self.class_combo = QComboBox()
        # for key, value in self.cls_dict.items():
        #     self.class_combo.addItem(value, key)
        # self.class_combo.setCurrentIndex(1)  # Set default to "Crater"
        # self.class_combo.currentIndexChanged.connect(self.update_current_class)

        class_layout = QHBoxLayout()
        for key, value in self.cls_dict.items():
            button = QPushButton(value)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, k=key: self.update_current_class(k))
            self.class_buttons[key] = button
            class_layout.addWidget(button)

        # Set the initial selected class
        self.update_current_class(self.current_class)

        left_layout.addSpacing(20)  # Add 20 pixels of vertical space

        left_layout.addWidget(QLabel("Select Class:"))
        left_layout.addLayout(class_layout)

        self.draw_mode_button = QPushButton("Draw Mode: Box 📦")
        self.draw_mode_button.clicked.connect(self.toggle_draw_mode)
        left_layout.addWidget(self.draw_mode_button)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()

        # Create the buttons
        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.undo)
        undo_button.setStyleSheet(
            """
            QPushButton {
                background-color: #FFB3BA;  /* Light red color */
                
                color: white;
                border: none;
                padding: 5px 10px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 4px;
            }
        """
        )

        redo_button = QPushButton("Redo")
        redo_button.clicked.connect(self.redo)
        redo_button.setStyleSheet(
            """
            QPushButton {
                background-color: #90EE90;  /* Light green color */
                color: black;
                border: none;
                padding: 5px 10px;
                text-align: center;
                text-decoration: none;
                margin: 4px 2px;
                border-radius: 4px;
            }
        """
        )

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_annotations)
        reset_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;  /* Red color */
                color: white;
                border: none;
                padding: 5px 10px;
                text-align: center;
                text-decoration: none;
                margin: 4px 2px;
                border-radius: 4px;
            }
        """
        )

        # Add the buttons to the horizontal layout
        button_layout.addWidget(undo_button)
        button_layout.addWidget(redo_button)
        button_layout.addWidget(reset_button)

        # Add the horizontal layout to the main left layout
        left_layout.addLayout(button_layout)

        self.visibility_button = QPushButton("Hide Annotations")
        self.visibility_button.clicked.connect(self.toggle_annotation_visibility)
        left_layout.addWidget(self.visibility_button)

        left_layout.addSpacing(20)  # Add 20 pixels of vertical space

        self.delete_all_points_button = QPushButton("Delete All Points")
        self.delete_all_points_button.clicked.connect(self.delete_all_points)
        self.delete_all_points_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;  /* Red color */
                color: white;
                border: none;
                padding: 5px 10px;
                text-align: center;
                text-decoration: none;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d32f2f;  /* Darker red for hover effect */
            }
            QPushButton:pressed {
                background-color: #b71c1c;  /* Even darker red when pressed */
            }
        """
        )
        left_layout.addWidget(self.delete_all_points_button)

        left_layout.addSpacing(20)  # Add 20 pixels of vertical space

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        left_layout.addWidget(QLabel("Confidence Threshold:"))
        left_layout.addWidget(self.confidence_slider)

        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.predict_with_loading)
        predict_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;  /* Green color */
                color: white;
                border: none;
                padding: 5px 10px;
                text-align: center;
                text-decoration: none;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;  /* Darker green for hover effect */
            }
            QPushButton:pressed {
                background-color: #3e8e41;  /* Even darker green when pressed */
            }
        """
        )
        left_layout.addWidget(predict_button)

        left_layout.addSpacing(20)  # Add 20 pixels of vertical space

        save_ann_button = QPushButton("Save Annotations")
        save_ann_button.clicked.connect(self.save_annotations)
        left_layout.addWidget(save_ann_button)

        layout.addWidget(left_panel, 1)

        # Right panel for image display and annotation
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        self.graphics_view = CustomGraphicsView()
        self.graphics_view.boxDrawn.connect(self.add_bounding_box)
        self.graphics_view.pointDrawn.connect(self.add_point)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        right_layout.addWidget(self.graphics_view)

        # self.attention_view = QLabel()
        # self.attention_view.setAlignment(Qt.AlignCenter)
        # right_layout.addWidget(self.attention_view)

        layout.addWidget(right_panel, 3)

    def on_confidence_changed(self):
        # Debounce the slider value changes
        self.debounce_timer.start(100)  # 100ms debounce time

    def update_filtered_predictions(self):
        if self.current_image in self.predicted_boxes:
            self.draw_annotations()

    def update_current_class(self, class_key):
        self.current_class = class_key
        for key, button in self.class_buttons.items():
            if key == class_key:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: black;
                        color: white;
                        border: none;
                        padding: 5px 10px;
                        text-align: center;
                        text-decoration: none;
                        font-size: 14px;
                        margin: 4px 2px;
                        border-radius: 4px;
                    }
                """
                )
                button.setChecked(True)
            else:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: white;
                        color: black;
                        border: 1px solid black;
                        padding: 5px 10px;
                        text-align: center;
                        text-decoration: none;
                        font-size: 14px;
                        margin: 4px 2px;
                        border-radius: 4px;
                    }
                """
                )
                button.setChecked(False)

    def load_images(self):
        self.image_directory = QFileDialog.getExistingDirectory(
            self, "Select Image Directory"
        )
        if self.image_directory:
            image_files = [
                f
                for f in os.listdir(self.image_directory)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                )
            ]
            self.image_list.clear()
            self.image_list.addItems(image_files)
            self.fetch_existing_annotations(self.image_directory)
            if image_files:
                self.load_image(self.image_list.item(0))

    def fetch_existing_annotations(self, directory):
        # In a real application, you'd fetch this from a server
        annotation_file = os.path.join(directory, "annotations.json")
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                self.bounding_boxes = json.load(f)

    def load_image(self, item):
        if isinstance(item, str):
            self.current_image = item
        else:
            self.current_image = item.text()

        image_path = os.path.join(self.image_directory, self.current_image)
        if os.path.exists(image_path):
            image = QImage(image_path)
            if not image.isNull():
                for item in self.annotation_items:
                    self.scene.removeItem(item)
                self.annotation_items.clear()

                pixmap = QPixmap.fromImage(image)
                self.scene.clear()
                self.scene.addPixmap(pixmap)
                self.graphics_view.updateSceneRect()
                self.graphics_view.fitInView(
                    self.scene.itemsBoundingRect(), Qt.KeepAspectRatio
                )
                self.draw_annotations()
                self.load_attention_mask()
            else:
                print(f"Failed to load image: {image_path}")
        else:
            print(f"Image file not found: {image_path}")

    def toggle_draw_mode(self):
        if self.graphics_view.drawMode == "box":
            self.graphics_view.drawMode = "point"
            self.draw_mode_button.setText("Draw Mode: Point 📍")
        else:
            self.graphics_view.drawMode = "box"
            self.draw_mode_button.setText("Draw Mode: Box 📦")

    def add_bounding_box(self, rect):
        if self.current_image:
            if self.current_image not in self.bounding_boxes:
                self.bounding_boxes[self.current_image] = []
            new_box = {
                "x": rect.x(),
                "y": rect.y(),
                "width": rect.width(),
                "height": rect.height(),
                "type": "box",
                "class": self.current_class,
            }
            self.bounding_boxes[self.current_image].append(new_box)
            self.undoStack.append(("add", self.current_image, new_box))
            self.redoStack.clear()
            self.draw_annotations()
            self.submit_annotation()

    def add_point(self, point):
        if self.current_image:
            if self.current_image not in self.bounding_boxes:
                self.bounding_boxes[self.current_image] = []
            new_point = {
                "x": point.x(),
                "y": point.y(),
                "type": "point",
                "class": self.current_class,
            }
            self.bounding_boxes[self.current_image].append(new_point)
            self.undoStack.append(("add", self.current_image, new_point))
            self.redoStack.clear()
            self.draw_annotations()
            self.submit_annotation()

    def draw_annotations(self):
        for item in self.annotation_items:
            self.scene.removeItem(item)
        self.annotation_items.clear()

        if self.current_image in self.bounding_boxes:
            for box in self.bounding_boxes[self.current_image]:
                color = Qt.red if box["class"] == 0 else Qt.green
                if box["type"] == "box":
                    rect = CustomGraphicsRectItem(
                        QRectF(box["x"], box["y"], box["width"], box["height"]), box
                    )
                    rect.setPen(QPen(QColor(color), 2))
                    self.scene.addItem(rect)
                    self.annotation_items.append(rect)

                    # Add delete button
                    delete_button = QPushButton("❌")
                    delete_button.setFixedSize(20, 20)
                    delete_button.clicked.connect(
                        lambda _, r=rect: self.delete_bounding_box(r)
                    )
                    delete_proxy = QGraphicsProxyWidget()
                    delete_proxy.setWidget(delete_button)
                    delete_proxy.setPos(box["x"] + box["width"] - 20, box["y"] - 20)
                    self.scene.addItem(delete_proxy)
                    self.annotation_items.append(delete_proxy)

                elif box["type"] == "point":
                    point_items = self.graphics_view.drawPoint(
                        QPointF(box["x"], box["y"])
                    )
                    for item in point_items:
                        item.setPen(QPen(QColor(color), 2))
                    self.annotation_items.extend(point_items)

        if self.current_image in self.predicted_boxes:
            confidence = self.confidence_slider.value() / 100
            for box in self.predicted_boxes[self.current_image]:
                if box["score"] >= confidence:
                    rect = CustomGraphicsRectItem(
                        QRectF(box["x"], box["y"], box["width"], box["height"]), box
                    )
                    rect.setPen(QPen(QColor(Qt.blue), 2))
                    self.scene.addItem(rect)
                    self.annotation_items.append(rect)

                    # Add accept button
                    accept_button = QPushButton("✔️")
                    accept_button.setFixedSize(20, 20)
                    accept_button.clicked.connect(
                        lambda _, r=rect: self.accept_prediction(r)
                    )
                    accept_proxy = QGraphicsProxyWidget()
                    accept_proxy.setWidget(accept_button)
                    accept_proxy.setPos(box["x"] + box["width"] - 40, box["y"] - 20)
                    self.scene.addItem(accept_proxy)
                    self.annotation_items.append(accept_proxy)

                    # Add reject button
                    reject_button = QPushButton("❌")
                    reject_button.setFixedSize(20, 20)
                    reject_button.clicked.connect(
                        lambda _, r=rect: self.reject_prediction(r)
                    )
                    reject_proxy = QGraphicsProxyWidget()
                    reject_proxy.setWidget(reject_button)
                    reject_proxy.setPos(box["x"] + box["width"] - 20, box["y"] - 20)
                    self.scene.addItem(reject_proxy)
                    self.annotation_items.append(reject_proxy)

                    # Add score text
                    score_text = self.scene.addText(f"🎯: {box['score']:.2f}")
                    score_text.setPos(box["x"] + box["width"] + 5, box["y"] - 20)
                    self.annotation_items.append(score_text)

    def delete_bounding_box(self, rect_item):
        box = rect_item.box_data
        if self.current_image in self.bounding_boxes:
            self.bounding_boxes[self.current_image] = [
                b for b in self.bounding_boxes[self.current_image] if b != box
            ]
            self.draw_annotations()

    def accept_prediction(self, rect_item):
        box = rect_item.box_data
        print(f"Accepting prediction: {box}")  # Debug print
        if self.current_image in self.predicted_boxes:
            # Remove the box from predicted_boxes
            self.predicted_boxes[self.current_image] = [
                b for b in self.predicted_boxes[self.current_image] if b != box
            ]

            # Add the box to bounding_boxes
            if self.current_image not in self.bounding_boxes:
                self.bounding_boxes[self.current_image] = []
            new_box = box.copy()
            new_box["type"] = "box"
            self.bounding_boxes[self.current_image].append(new_box)

            # Redraw annotations
            self.draw_annotations()

    def reject_prediction(self, rect_item):
        box = rect_item.box_data
        if self.current_image in self.predicted_boxes:
            # Remove the box from predicted_boxes
            self.predicted_boxes[self.current_image] = [
                b for b in self.predicted_boxes[self.current_image] if b != box
            ]

            # Redraw annotations
            self.draw_annotations()

    def predict_with_loading(self):
        if not self.current_image:
            return

        # Create and show the progress dialog
        # Create progress dialog on the main thread
        QTimer.singleShot(
            0,
            lambda: self.progress_manager.create_progress_dialog(0, 0, "Preparing..."),
        )

        # Start the preparation process
        QTimer.singleShot(0, self.perform_prediction)

        # Start the prediction in a separate thread
        # self.perform_prediction()

    def is_thread_running(self):
        """Safely check if the prediction thread is running."""
        if self.prediction_thread is None:
            return False
        try:
            # Check if the C++ object still exists
            if sip.isdeleted(self.prediction_thread):
                self.prediction_thread = None
                return False
            return self.prediction_thread.isRunning()
        except RuntimeError:
            # The QThread object has been deleted
            self.prediction_thread = None
            return False

    def perform_prediction(self):
        if self.is_thread_running():
            print("Prediction is already running")
            return

        image_path = os.path.join(self.image_directory, self.current_image)
        bboxes = self.bounding_boxes.get(self.current_image, [])

        self.prediction_thread = QThread()
        self.prediction_worker = PredictionWorker(self.predictor, image_path, bboxes)
        self.prediction_worker.moveToThread(self.prediction_thread)
        self.prediction_thread.started.connect(self.prediction_worker.run)
        self.prediction_worker.finished.connect(self.prediction_thread.quit)
        self.prediction_worker.finished.connect(self.prediction_worker.deleteLater)
        self.prediction_thread.finished.connect(self.prediction_thread.deleteLater)
        self.prediction_worker.finished.connect(self.prediction_complete)
        self.prediction_worker.progress.connect(self.update_progress)

        # Create progress dialog
        self.progress_manager.create_progress_dialog(0, 100, "Processing...")

        self.prediction_thread.start()

    @pyqtSlot(float)
    def update_progress(self, progress):
        # Convert progress to percentage (0-100)
        progress_percent = int(progress * 100)
        self.progress_manager.update_progress.emit(
            progress_percent, 100, "Processing patches..."
        )

    @pyqtSlot(object, object, object)
    def prediction_complete(self, selected_boxes, selected_scores, selected_classes):
        print("Main: Prediction complete")  # Debug print
        self.progress_manager.close_progress_dialog()

        if selected_boxes is None:
            self.statusBar().showMessage("Prediction cancelled", 3000)
        else:
            self.process_predictions(selected_boxes, selected_scores, selected_classes)
            self.draw_annotations()
            self.statusBar().showMessage("Prediction complete", 3000)

    @staticmethod
    def calculate_iou(box1, box2):
        # Calculate the (x, y)-coordinates of the intersection rectangle
        x_left = max(box1["x"], box2["x"])
        y_top = max(box1["y"], box2["y"])
        x_right = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y_bottom = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate the area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate the area of both bounding boxes
        box1_area = box1["width"] * box1["height"]
        box2_area = box2["width"] * box2["height"]

        # Calculate the IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def process_predictions(self, selected_boxes, selected_scores, selected_classes):
        self.predicted_boxes[self.current_image] = []

        # Get the dimensions of the current image
        pixmap = self.scene.items()[
            -1
        ].pixmap()  # Assuming the pixmap is the last item in the scene
        image_width = pixmap.width()
        image_height = pixmap.height()

        for pred, score, cls in zip(selected_boxes, selected_scores, selected_classes):
            # Convert relative coordinates to absolute pixel coordinates
            center_x = pred[0] * image_width
            center_y = pred[1] * image_height
            width = pred[2] * image_width
            height = pred[3] * image_height

            x_top_left = center_x - width / 2
            y_top_left = center_y - height / 2

            rounded_score = round(score, 2)

            predicted_box = {
                "x": x_top_left,
                "y": y_top_left,
                "width": width,
                "height": height,
                "score": rounded_score,
                "class": cls,
            }

            # Check overlap with existing normal boxes
            overlap_threshold = 0.95
            should_add = True

            for normal_box in self.bounding_boxes.get(self.current_image, []):
                if (
                    normal_box["type"] == "box"
                ):  # Ensure we're only comparing with box annotations, not points
                    iou = self.calculate_iou(predicted_box, normal_box)
                    if iou > overlap_threshold:
                        should_add = False
                        break

                # check if the box is within the image
                if (
                    x_top_left < 0
                    or y_top_left < 0
                    or x_top_left + width > image_width
                    or y_top_left + height > image_height
                ):
                    should_add = False
                    break

                # check if the box is relatively square
                if width / height > 2 or height / width > 2:
                    should_add = False
                    break

            if should_add:
                self.predicted_boxes[self.current_image].append(predicted_box)

    def delete_all_points(self):
        if self.current_image in self.bounding_boxes:
            self.bounding_boxes[self.current_image] = [
                box
                for box in self.bounding_boxes[self.current_image]
                if box["type"] != "point"
            ]
        self.draw_annotations()

    def undo(self):
        if self.undoStack:
            action = self.undoStack.pop()
            self.redoStack.append(action)
            if action[0] == "add":
                self.bounding_boxes[action[1]].remove(action[2])
            elif action[0] == "remove":
                self.bounding_boxes[action[1]].append(action[2])
            self.draw_annotations()

    def redo(self):
        if self.redoStack:
            action = self.redoStack.pop()
            self.undoStack.append(action)
            if action[0] == "add":
                self.bounding_boxes[action[1]].append(action[2])
            elif action[0] == "remove":
                self.bounding_boxes[action[1]].remove(action[2])
            self.draw_annotations()

    def reset_annotations(self):
        if self.current_image:
            self.undoStack.append(
                (
                    "remove_all",
                    self.current_image,
                    self.bounding_boxes[self.current_image][:],
                )
            )
            self.bounding_boxes[self.current_image] = []
            self.redoStack.clear()
            self.draw_annotations()

    def toggle_annotation_visibility(self):
        for item in self.scene.items():
            if isinstance(item, (QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsProxyWidget, QGraphicsTextItem)):
                item.setVisible(not item.isVisible())

        if self.visibility_button.text() == "Hide Annotations":
            self.visibility_button.setText("Show Annotations")
        else:
            self.visibility_button.setText("Hide Annotations")

    def filter_predictions(self):
        self.draw_annotations()

    def submit_annotation(self):
        if self.current_image and self.current_image in self.bounding_boxes:
            print(
                f"Submitting annotation for {self.current_image}: {self.bounding_boxes[self.current_image]}"
            )
            # Here you would typically send this data to a server or save it to a file

    def load_attention_mask(self):
        attention_mask_path = os.path.join(
            self.image_directory, "attn_masks", f"{self.current_image}_attn.png"
        )
        if os.path.exists(attention_mask_path):
            pixmap = QPixmap(attention_mask_path)
            self.attention_view.setPixmap(
                pixmap.scaled(
                    self.attention_view.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            return
            # self.attention_view.setText("Attention mask not available")

    def save_annotations(self):
        """Open menu to select save location and save currently displayed annotations."""
        if not self.bounding_boxes:
            print("No annotations to save")
            return

        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Annotations"
        )
        if not save_dir:
            print("No directory selected")
            return

        annotation_file = os.path.join(save_dir, "annotations.json")
        with open(annotation_file, "w") as f:
            json.dump(self.bounding_boxes, f, indent=4)
        print(f"Annotations saved to: {annotation_file}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image:
            self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # def closeEvent(self, event):
    #     if self.prediction_thread.isRunning():
    #         self.prediction_thread.quit()
    #         self.prediction_thread.wait()
    #     super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageAnnotator()
    ex.show()
    sys.exit(app.exec_())
