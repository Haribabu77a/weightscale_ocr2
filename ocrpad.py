import sys
import os
import time
import cv2
import numpy as np
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR Backend
# Install: pip install easyocr
# ─────────────────────────────────────────────────────────────────────────────
try:
    import easyocr
except ImportError:
    print("\n[ERROR] easyocr is not installed.")
    print("Run:  pip install easyocr\n")
    sys.exit(1)

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect, QPoint, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QSlider,
    QTextEdit, QGroupBox, QFormLayout, QScrollArea, QCheckBox,
    QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen


# ─────────────────────────────────────────────────────────────────────────────
# Custom Gap Measurement (The "11" Splicer)
# ─────────────────────────────────────────────────────────────────────────────

def remove_massive_gaps(binary_img, max_allowed_gap):
    """
    Scans the image column by column to MEASURE THE WIDTH of the empty gaps.
    If a gap is wider than 'max_allowed_gap', it cuts out the excess pixels.
    This safely pulls isolated numbers together without squishing them into a '4'.
    """
    # Invert so text is >0
    inv = cv2.bitwise_not(binary_img)
    col_sums = np.sum(inv, axis=0)
    has_text = col_sums > 0
    
    new_cols = []
    current_gap = 0
    
    for x in range(binary_img.shape[1]):
        if has_text[x]:
            if current_gap > 0:
                # Add the empty space back, but CAP IT at max_allowed_gap
                gap_to_add = min(current_gap, max_allowed_gap)
                for _ in range(gap_to_add):
                    new_cols.append(np.full(binary_img.shape[0], 255, dtype=np.uint8))
            
            current_gap = 0
            new_cols.append(binary_img[:, x])
        else:
            current_gap += 1
            
    if len(new_cols) == 0:
        return binary_img
        
    return np.column_stack(new_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Shared preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def compute_binary(frame_bgr, roi_rect, use_otsu: bool, manual_thresh: int, morph_iters: int, gap_limit: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    # ── 1. ROI crop ──────────────────────────────────────────────────────────
    if roi_rect and roi_rect.width() > 5 and roi_rect.height() > 5:
        x1 = max(0, min(roi_rect.x(), w - 1))
        y1 = max(0, min(roi_rect.y(), h - 1))
        x2 = max(x1 + 1, min(roi_rect.x() + roi_rect.width(),  w))
        y2 = max(y1 + 1, min(roi_rect.y() + roi_rect.height(), h))
        region = frame_bgr[y1:y2, x1:x2]
        if region.size == 0:
            region = frame_bgr
    else:
        region = frame_bgr

    # ── 2. Uniform Upscale ───────────────────────────────────────────────────
    region = cv2.resize(region, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # ── 3. Grayscale & Blur ──────────────────────────────────────────────────
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── 4. Binarize ──────────────────────────────────────────────────────────
    if use_otsu:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, manual_thresh, 255, cv2.THRESH_BINARY)

    # ── 5. Ensure Dark Text on White Background ──────────────────────────────
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # ── 6. PIXEL GAP SPLICING (Your Logic) ───────────────────────────────────
    binary = remove_massive_gaps(binary, max_allowed_gap=gap_limit)

    # ── 7. Morphological Closing ─────────────────────────────────────────────
    if morph_iters > 0:
        binary_inverted = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary_inverted, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
        binary = cv2.bitwise_not(closed)

    # ── 8. Pad & Final Anti-Aliasing ─────────────────────────────────────────
    padded = cv2.copyMakeBorder(binary, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    final_ready = cv2.GaussianBlur(padded, (3, 3), 0)

    return final_ready


# ─────────────────────────────────────────────────────────────────────────────
# OCR Worker
# ─────────────────────────────────────────────────────────────────────────────

class OCRWorker(QThread):
    result_ready = pyqtSignal(str, float, np.ndarray)

    def __init__(self):
        super().__init__()
        self.reader = easyocr.Reader(['en']) 
        
        self.frame  = None
        self.roi    = None
        self.config = {}
        self._busy  = False

        self._last_raw    = None
        self._last_binary = None

    def update_task(self, frame, roi, config):
        if self._busy:
            return
        self.frame  = frame.copy()
        self.roi    = roi
        self.config = config.copy()
        self._busy  = True
        self.start()

    def refilter(self, confidence: float):
        if self._last_raw is None or self._last_binary is None:
            return
        lines = self._parse_result(self._last_raw, confidence)
        text  = "\n".join(lines) if lines else "No text detected."
        self.result_ready.emit(text, 0.0, self._last_binary.copy())

    def run(self):
        if self.frame is None:
            self._busy = False
            return
        try:
            start_time = time.time()

            binary = compute_binary(
                self.frame,
                self.roi,
                self.config.get('use_otsu', True),
                self.config.get('manual_thresh', 127),
                self.config.get('morph_iters', 0),
                self.config.get('gap_limit', 50)
            )

            ocr_input = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            raw = self.reader.readtext(
                ocr_input, 
                allowlist='0123456789.', 
                width_ths=2.0,       
                text_threshold=0.1,  
                low_text=0.1         
            )

            self._last_raw    = raw
            self._last_binary = binary.copy()

            conf  = self.config.get('confidence', 0.15)
            lines = self._parse_result(raw, conf)
            text  = "\n".join(lines) if lines else "No text detected."
            elapsed_ms = (time.time() - start_time) * 1000

            self.result_ready.emit(text, elapsed_ms, binary.copy())

        except Exception:
            tb = traceback.format_exc()
            print("OCR Worker Error:\n", tb)
            blank = np.zeros((100, 100), dtype=np.uint8)
            self.result_ready.emit(f"Error during OCR:\n{tb}", 0.0, blank)
        finally:
            self._busy = False

    @staticmethod
    def _parse_result(result, confidence: float = 0.15) -> list:
        if not result:
            return []
        
        valid_items = []
        for item in result:
            try:
                text  = str(item[1]).replace(" ", "")
                score = float(item[2])
                x_pos = float(item[0][0][0]) 
                
                if text and score >= confidence:
                    valid_items.append((x_pos, text, score))
            except (IndexError, TypeError, ValueError):
                continue
                
        if not valid_items:
            return []
            
        valid_items.sort(key=lambda x: x[0])
        combined_text = "".join([i[1] for i in valid_items])
        
        if len(valid_items) > 0:
            avg_score = sum([i[2] for i in valid_items]) / len(valid_items)
        else:
            avg_score = 0.0
            
        return [f"{combined_text}   [{avg_score:.2f}]"]


# ─────────────────────────────────────────────────────────────────────────────
# Video / Camera Thread
# ─────────────────────────────────────────────────────────────────────────────

class VideoThread(QThread):
    frame_ready  = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap     = None
        self.running = False
        self.source  = None

    def set_source(self, source):
        self.source = source

    def run(self):
        if self.source is None:
            return
        self.running = True

        if isinstance(self.source, int) and os.name == 'nt':
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.error_signal.emit(
                f"Could not open source: {self.source}\n"
                "Check the camera is connected and not used by another app.")
            self.running = False
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.error_signal.emit("Camera feed lost.")
                    break
            self.frame_ready.emit(frame)
            time.sleep(0.033)

        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.running = False
        if self.isRunning():
            self.wait()


# ─────────────────────────────────────────────────────────────────────────────
# Video label with ROI drawing
# ─────────────────────────────────────────────────────────────────────────────

class VideoLabel(QLabel):
    roi_changed = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.start_point      = QPoint()
        self.end_point        = QPoint()
        self.is_drawing       = False
        self.roi_mode_enabled = False
        self.current_roi      = QRect()
        self.setMouseTracking(True)

    def set_roi_mode(self, enabled):
        self.roi_mode_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if self.roi_mode_enabled and event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point   = event.pos()
            self.is_drawing  = True

    def mouseMoveEvent(self, event):
        if self.roi_mode_enabled and self.is_drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.roi_mode_enabled and event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing  = False
            self.current_roi = QRect(self.start_point, self.end_point).normalized()
            self.roi_changed.emit(self.current_roi)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        if self.is_drawing:
            painter.drawRect(QRect(self.start_point, self.end_point).normalized())
        elif not self.current_roi.isEmpty():
            painter.drawRect(self.current_roi)

    def clear_roi(self):
        self.current_roi = QRect()
        self.update()
        self.roi_changed.emit(self.current_roi)


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Pipeline — Custom Gap Splicing")
        self.setGeometry(100, 100, 1260, 950)

        self.current_frame   = None
        self.selected_roi    = None
        self.continuous_mode = False

        self._ocr_debounce = QTimer(singleShot=True, interval=350)
        self._ocr_debounce.timeout.connect(self.trigger_single)

        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.error_signal.connect(self.show_error)

        self.ocr_worker = OCRWorker()
        self.ocr_worker.result_ready.connect(self.display_ocr_results)

        self._init_ui()

    def _init_ui(self):
        main_layout  = QHBoxLayout()
        left_layout  = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedSize(660, 500)

        self.view_feed = VideoLabel()
        self.view_feed.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.view_feed.setStyleSheet("background-color: black;")
        self.view_feed.roi_changed.connect(self.handle_roi_update)
        self.scroll_area.setWidget(self.view_feed)
        left_layout.addWidget(self.scroll_area)

        thumb_row = QHBoxLayout()
        thumb_col = QVBoxLayout()
        thumb_col.addWidget(QLabel("Preprocessed (what OCR sees):"))
        self.view_preprocessed = QLabel()
        self.view_preprocessed.setFixedSize(320, 240)
        self.view_preprocessed.setStyleSheet("background-color: #222; color: #aaa;")
        self.view_preprocessed.setAlignment(Qt.AlignCenter)
        self.view_preprocessed.setText("Run OCR to see preview")
        thumb_col.addWidget(self.view_preprocessed)
        thumb_row.addLayout(thumb_col)
        thumb_row.addStretch()
        left_layout.addLayout(thumb_row)

        group_media = QGroupBox("Input Source")
        mf = QFormLayout()
        self.combo_source = QComboBox()
        self.combo_source.addItems(["Internal Camera (0)", "USB Camera (1)"])
        mf.addRow("Camera:", self.combo_source)
        btn_cam = QPushButton("Start Selected Camera")
        btn_cam.clicked.connect(self.handle_source_change)
        mf.addRow("", btn_cam)
        btn_upload = QPushButton("Upload Image / Video File")
        btn_upload.clicked.connect(self.handle_file_upload)
        mf.addRow("File:", btn_upload)
        group_media.setLayout(mf)
        right_layout.addWidget(group_media)

        # Binarization
        group_pre = QGroupBox("Binarization")
        pre_layout = QVBoxLayout()

        self.chk_otsu = QCheckBox("Auto Otsu (recommended)")
        self.chk_otsu.setChecked(True)
        self.chk_otsu.stateChanged.connect(self._on_otsu_toggled)
        pre_layout.addWidget(self.chk_otsu)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Manual Threshold:"))
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(127)
        self.slider_thresh.setEnabled(False)
        self.slider_thresh.valueChanged.connect(self._on_thresh_changed)
        thresh_row.addWidget(self.slider_thresh)
        self.lbl_thresh_val = QLabel("127")
        self.lbl_thresh_val.setFixedWidth(30)
        thresh_row.addWidget(self.lbl_thresh_val)
        pre_layout.addLayout(thresh_row)
        
        group_pre.setLayout(pre_layout)
        right_layout.addWidget(group_pre)

        # The "11" Fix Splicer Box
        group_splicer = QGroupBox("Gap Splicing (The '11' Fix)")
        splicer_layout = QVBoxLayout()
        
        gap_row = QHBoxLayout()
        gap_row.addWidget(QLabel("Max Gap Limit:"))
        self.slider_gap = QSlider(Qt.Horizontal)
        self.slider_gap.setRange(10, 100)
        self.slider_gap.setValue(50)  # Safe default to prevent 11s from squishing
        self.slider_gap.setTickInterval(10)
        self.slider_gap.setTickPosition(QSlider.TicksBelow)
        self.slider_gap.valueChanged.connect(self._on_gap_changed)
        gap_row.addWidget(self.slider_gap)
        self.lbl_gap_val = QLabel("50")
        self.lbl_gap_val.setFixedWidth(30)
        gap_row.addWidget(self.lbl_gap_val)
        splicer_layout.addLayout(gap_row)
        
        gap_hint = QLabel("If '1' vanishes: LOWER limit. If '11' becomes '4': INCREASE limit.")
        gap_hint.setStyleSheet("color: #cc0000; font-size: 11px; font-weight: bold;")
        splicer_layout.addWidget(gap_hint)
        
        group_splicer.setLayout(splicer_layout)
        right_layout.addWidget(group_splicer)

        # Morphology
        group_morph = QGroupBox("Morphology (Optional)")
        morph_layout = QVBoxLayout()
        
        morph_row = QHBoxLayout()
        morph_row.addWidget(QLabel("Closing Amount:"))
        self.slider_morph = QSlider(Qt.Horizontal)
        self.slider_morph.setRange(0, 5)
        self.slider_morph.setValue(0)
        self.slider_morph.setTickInterval(1)
        self.slider_morph.setTickPosition(QSlider.TicksBelow)
        self.slider_morph.valueChanged.connect(self._on_morph_changed)
        morph_row.addWidget(self.slider_morph)
        self.lbl_morph_val = QLabel("0")
        self.lbl_morph_val.setFixedWidth(30)
        morph_row.addWidget(self.lbl_morph_val)
        morph_layout.addLayout(morph_row)
        
        group_morph.setLayout(morph_layout)
        right_layout.addWidget(group_morph)

        # Confidence Filter
        group_conf = QGroupBox("OCR Confidence Filter")
        conf_layout = QVBoxLayout()

        self.lbl_conf = QLabel("Min Confidence: 0.15")
        self.lbl_conf.setStyleSheet("font-weight: bold;")
        conf_layout.addWidget(self.lbl_conf)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("0.00"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(15)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.valueChanged.connect(self._on_confidence_changed)
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(QLabel("1.00"))
        conf_layout.addLayout(conf_row)

        group_conf.setLayout(conf_layout)
        right_layout.addWidget(group_conf)

        # Execution Controls
        group_actions = QGroupBox("Execution Controls")
        act = QVBoxLayout()

        roi_row = QHBoxLayout()
        self.btn_toggle_roi = QPushButton("Enable ROI Selection Mode")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.clicked.connect(self.toggle_roi_mode)
        btn_roi_clear = QPushButton("Reset ROI")
        btn_roi_clear.clicked.connect(self.clear_roi)
        roi_row.addWidget(self.btn_toggle_roi)
        roi_row.addWidget(btn_roi_clear)
        act.addLayout(roi_row)

        btn_row = QHBoxLayout()
        self.btn_single = QPushButton("Single Run")
        self.btn_auto   = QPushButton("Continuous Run")
        self.btn_stop   = QPushButton("Stop")
        self.btn_single.clicked.connect(self.trigger_single)
        self.btn_auto.clicked.connect(self.trigger_auto)
        self.btn_stop.clicked.connect(self.trigger_stop)
        btn_row.addWidget(self.btn_single)
        btn_row.addWidget(self.btn_auto)
        btn_row.addWidget(self.btn_stop)
        act.addLayout(btn_row)
        group_actions.setLayout(act)
        right_layout.addWidget(group_actions)

        self.lbl_timer = QLabel("Inference Delay: — ms")
        self.lbl_timer.setStyleSheet("font-weight: bold; color: #0055cc;")
        right_layout.addWidget(self.lbl_timer)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setStyleSheet("font-size: 13px; font-family: monospace;")
        right_layout.addWidget(self.text_output)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        w = QWidget()
        w.setLayout(main_layout)
        self.setCentralWidget(w)

    def _on_otsu_toggled(self, state):
        self.slider_thresh.setEnabled(not bool(state))
        self._refresh_preview_and_queue_ocr()

    def _on_thresh_changed(self, value):
        self.lbl_thresh_val.setText(str(value))
        self._refresh_preview_and_queue_ocr()

    def _on_morph_changed(self, value):
        self.lbl_morph_val.setText(str(value))
        self._refresh_preview_and_queue_ocr()
        
    def _on_gap_changed(self, value):
        self.lbl_gap_val.setText(str(value))
        self._refresh_preview_and_queue_ocr()

    def _refresh_preview_and_queue_ocr(self):
        if self.current_frame is None:
            return
        binary = compute_binary(
            self.current_frame,
            self.selected_roi,
            self.chk_otsu.isChecked(),
            self.slider_thresh.value(),
            self.slider_morph.value(),
            self.slider_gap.value()
        )
        self._show_preview(binary)
        self._ocr_debounce.start()

    def _on_confidence_changed(self, value):
        conf = value / 100.0
        self.lbl_conf.setText(f"Min Confidence: {conf:.2f}")
        self.ocr_worker.refilter(conf)

    def handle_source_change(self):
        self.video_thread.stop()
        self.video_thread.set_source(self.combo_source.currentIndex())
        self.video_thread.start()

    def handle_file_upload(self):
        self.video_thread.stop()
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Media", "",
            "Image / Video Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov)")
        if not path:
            return
        if path.lower().endswith(('.mp4', '.avi', '.mov')):
            self.video_thread.set_source(path)
            self.video_thread.start()
        else:
            img = cv2.imread(path)
            if img is not None:
                self.update_frame(img)
            else:
                self.show_error(f"Could not read image:\n{path}")

    def toggle_roi_mode(self, checked):
        self.btn_toggle_roi.setText(
            "ROI Mode Active — Draw a Box" if checked else "Enable ROI Selection Mode")
        self.view_feed.set_roi_mode(checked)

    def clear_roi(self):
        self.view_feed.clear_roi()
        self.selected_roi = None
        self._refresh_preview_and_queue_ocr()

    def handle_roi_update(self, rect):
        self.selected_roi = rect if not rect.isEmpty() else None
        self.btn_toggle_roi.setChecked(False)
        self.toggle_roi_mode(False)
        self._refresh_preview_and_queue_ocr()

    def update_frame(self, frame):
        self.current_frame = frame
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img  = QImage(bytes(rgb.data), w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.view_feed.setPixmap(pixmap)
        self.view_feed.resize(pixmap.size())

        if self.continuous_mode:
            self.trigger_single()

    def trigger_single(self):
        if self.current_frame is None:
            return
        self.ocr_worker.update_task(
            self.current_frame,
            self.selected_roi,
            {
                'use_otsu':       self.chk_otsu.isChecked(),
                'manual_thresh':  self.slider_thresh.value(),
                'morph_iters':    self.slider_morph.value(),
                'gap_limit':      self.slider_gap.value(),
                'confidence':     self.conf_slider.value() / 100.0,
            }
        )

    def trigger_auto(self):
        self.continuous_mode = True
        self.btn_auto.setEnabled(False)
        self.btn_single.setEnabled(False)

    def trigger_stop(self):
        self.continuous_mode = False
        self.btn_auto.setEnabled(True)
        self.btn_single.setEnabled(True)

    def display_ocr_results(self, text, runtime, debug_img):
        if runtime > 0:
            self.lbl_timer.setText(f"Inference Delay: {runtime:.2f} ms")
        self.text_output.setPlainText(text)
        self._show_preview(debug_img)

    def _show_preview(self, binary: np.ndarray):
        h, w = binary.shape
        q_img = QImage(bytes(binary.data), w, h, w, QImage.Format_Grayscale8)
        self.view_preprocessed.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_error(self, msg: str):
        QMessageBox.warning(self, "Error", msg)

    def closeEvent(self, event):
        self.continuous_mode = False
        self._ocr_debounce.stop()
        self.video_thread.stop()
        if self.ocr_worker.isRunning():
            self.ocr_worker.terminate()
            self.ocr_worker.wait()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
