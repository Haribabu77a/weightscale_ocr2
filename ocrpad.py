import sys
import os
import time
import cv2
import numpy as np
import re
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
    QMessageBox, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen


# ─────────────────────────────────────────────────────────────────────────────
# Custom Gap Measurement (The "11" Splicer) — LEGACY PATH ONLY
# ─────────────────────────────────────────────────────────────────────────────

def remove_massive_gaps(binary_img, max_allowed_gap):
    """
    Scans column by column and caps empty gaps at 'max_allowed_gap'.
    NOTE: only used by the legacy CRAFT path.
    """
    inv = cv2.bitwise_not(binary_img)
    col_sums = np.sum(inv, axis=0)
    has_text = col_sums > 0

    new_cols = []
    current_gap = 0

    for x in range(binary_img.shape[1]):
        if has_text[x]:
            if current_gap > 0:
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
# Shared preprocessing — (Biased Otsu -> Area Filter -> Closing)
# ─────────────────────────────────────────────────────────────────────────────

def compute_binary(frame_bgr, roi_rect, use_otsu: bool, otsu_offset: int, manual_thresh: int,
                   area_threshold: int, morph_iters: int, gap_limit: int, do_splice: bool = True) -> np.ndarray:
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

    # ── 4. Binarize (Biased Otsu Implementation) ─────────────────────────────
    if use_otsu:
        # Step 1: Let Otsu calculate the mathematical ideal threshold
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Step 2: Apply the user's bias offset (e.g. +5) and clamp between 0-255
        final_thresh = max(0, min(255, int(otsu_val) + otsu_offset))
        # Step 3: Run the actual binarization with our biased value
        _, binary = cv2.threshold(blurred, final_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(blurred, manual_thresh, 255, cv2.THRESH_BINARY)

    # ── 5. Ensure Dark Text on White Background ──────────────────────────────
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # ── 6. PIXEL GAP SPLICING (legacy CRAFT path only) ───────────────────────
    if do_splice:
        binary = remove_massive_gaps(binary, max_allowed_gap=gap_limit)

    # ── 7. AREA FILTER (Surgically deletes small dots FIRST) ─────────────────
    if area_threshold > 0:
        binary_inverted = cv2.bitwise_not(binary)
        
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inverted, connectivity=8)
        clean_mask = np.zeros_like(binary_inverted)
        
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= area_threshold:
                clean_mask[labels == i] = 255
                
        binary = cv2.bitwise_not(clean_mask)

    # ── 8. MORPHOLOGICAL CLOSING (Heals broken LCD segments SECOND) ──────────
    if morph_iters > 0:
        binary_inverted = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_digits = cv2.morphologyEx(binary_inverted, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
        binary = cv2.bitwise_not(closed_digits)

    # ── 9. Pad & Final Anti-Aliasing ─────────────────────────────────────────
    padded = cv2.copyMakeBorder(binary, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    final_ready = cv2.GaussianBlur(padded, (3, 3), 0)

    return final_ready


# ─────────────────────────────────────────────────────────────────────────────
# Per-digit segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_digits(binary,
                   min_h_frac: float = 0.40,
                   split_ratio: float = 1.5):
    
    text = cv2.bitwise_not(binary)                      
    _, text = cv2.threshold(text, 127, 255, cv2.THRESH_BINARY)

    num, _labels, stats, _cent = cv2.connectedComponentsWithStats(text, connectivity=8)
    if num <= 1:
        return []

    raw = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        raw.append((x, y, ww, hh))

    if not raw:
        return []

    max_h = max(c[3] for c in raw)

    digits = []
    for (x, y, ww, hh) in raw:
        if hh >= min_h_frac * max_h:
            digits.append([x, y, ww, hh])

    if not digits:
        return []

    digits.sort(key=lambda c: c[0])

    widths = sorted(c[2] for c in digits)
    med = widths[len(widths) // 2]

    split = []
    for (x, y, ww, hh) in digits:
        if med <= 0 or ww <= split_ratio * med:
            split.append([x, y, ww, hh])
            continue

        sub = text[y:y + hh, x:x + ww]
        cuts = _gap_cuts(sub, min_sub=max(3, int(0.45 * med)))

        if cuts:
            prev = 0
            for c in list(cuts) + [ww]:
                if c - prev > 0:
                    split.append([x + prev, y, c - prev, hh])
                prev = c
        elif ww >= 1.9 * med and len(digits) >= 3:
            n = max(2, int(round(ww / med)))
            sw = ww / n
            for k in range(n):
                split.append([int(x + k * sw), y, int(round(sw)), hh])
        else:
            split.append([x, y, ww, hh])

    digit_boxes = [[x, x + ww, y, y + hh] for (x, y, ww, hh) in split]
    return digit_boxes


def _gap_cuts(text_mask, min_sub: int, gap_frac: float = 0.06):
    h, w = text_mask.shape[:2]
    if w == 0 or h == 0:
        return []
    col_ink = (text_mask > 0).sum(axis=0).astype(float)
    thr = max(2.0, gap_frac * h)

    cuts = []
    last = 0
    i = min_sub
    while i < w - min_sub:
        if col_ink[i] <= thr:
            j = i
            while j < w and col_ink[j] <= thr:
                j += 1
            center = (i + j) // 2
            if center - last >= min_sub and (w - center) >= min_sub:
                cuts.append(center)
                last = center
            i = j
        else:
            i += 1
    return cuts


def annotate_segmentation(binary, digit_boxes):
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (x0, x1, y0, y1) in digit_boxes:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 170, 0), 2)
    return vis


def pad_digit_boxes(boxes, W, H, pad_x: float = 0.30, pad_y: float = 0.30,
                    gap_frac: float = 0.45, min_aspect_ratio: float = 0.60):
    boxes = sorted(boxes, key=lambda b: b[0])
    n = len(boxes)
    out = []
    
    for i, (x0, x1, y0, y1) in enumerate(boxes):
        bw, bh = x1 - x0, y1 - y0
        
        px = int(round(bw * pad_x))
        py = int(round(bh * pad_y))

        target_w = max(bw, int(bh * min_aspect_ratio))
        extra_w = max(0, target_w - bw)
        
        px_req_left = px + (extra_w // 2)
        px_req_right = px + (extra_w - (extra_w // 2))

        if i > 0:
            left_gap = x0 - boxes[i - 1][1]
            px_left = min(px_req_left, max(0, int(left_gap * gap_frac)))
        else:
            px_left = px_req_left
            
        if i < n - 1:
            right_gap = boxes[i + 1][0] - x1
            px_right = min(px_req_right, max(0, int(right_gap * gap_frac)))
        else:
            px_right = px_req_right

        out.append([max(0, x0 - px_left), min(W, x1 + px_right),
                    max(0, y0 - py), min(H, y1 + py)])
    return out


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

        self._last_mode   = 'perdigit'
        self._last_raw    = None
        self._last_binary = None
        self._last_text   = None
        self._last_vis    = None

    def update_task(self, frame, roi, config):
        if self._busy:
            return
        self.frame  = frame.copy()
        self.roi    = roi
        self.config = config.copy()
        self._busy  = True
        self.start()

    def refilter(self, confidence: float):
        if self._last_mode == 'perdigit':
            if self._last_text is not None and self._last_vis is not None:
                self.result_ready.emit(self._last_text, 0.0, self._last_vis.copy())
            return

        if self._last_raw is None or self._last_binary is None:
            return
        
        dec_pos = self.config.get('dec_pos', 0)
        lines = self._parse_result(self._last_raw, confidence, dec_pos)
        text  = "\n".join(lines) if lines else "No text detected."
        self.result_ready.emit(text, 0.0, self._last_binary.copy())

    def run(self):
        if self.frame is None:
            self._busy = False
            return
        try:
            start_time = time.time()
            if self.config.get('per_digit', True):
                self._run_per_digit()
            else:
                self._run_legacy()
        except Exception:
            tb = traceback.format_exc()
            print("OCR Worker Error:\n", tb)
            blank = np.zeros((100, 100), dtype=np.uint8)
            self.result_ready.emit(f"Error during OCR:\n{tb}", 0.0, blank)
        finally:
            self._busy = False

    # ── Per-digit pipeline ───────────────────────────────────────────────────
    def _run_per_digit(self):
        start_time = time.time()

        binary = compute_binary(
            self.frame, self.roi,
            self.config.get('use_otsu', True),
            self.config.get('otsu_offset', 5),
            self.config.get('manual_thresh', 127),
            self.config.get('area_threshold', 0),
            self.config.get('morph_iters', 0),
            self.config.get('gap_limit', 50),
            do_splice=False,
        )
        H, W = binary.shape
        pad = self.config.get('pad_pct', 30) / 100.0

        digit_boxes = segment_digits(binary)
        padded = pad_digit_boxes(digit_boxes, W, H, pad_x=pad, pad_y=pad)

        vis = annotate_segmentation(binary, padded)
        self._last_mode   = 'perdigit'
        self._last_binary = binary.copy()
        self._last_vis    = vis.copy()

        if not digit_boxes:
            self._last_text = "No digits detected."
            self.result_ready.emit(self._last_text, 0.0, vis)
            return

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        thick  = cv2.erode(binary, kernel, iterations=1)

        tokens = []
        confs  = []
        empties = 0
        for box, pbox in zip(sorted(digit_boxes, key=lambda b: b[0]), padded):
            ch, conf = self._read_digit_vote(binary, thick, pbox)
            if ch == '':
                empties += 1
                continue
            confs.append(conf)
            tokens.append(((box[0] + box[1]) / 2.0, ch))

        tokens.sort(key=lambda t: t[0])
        raw_number_str = ''.join(ch for _, ch in tokens)
        
        # Manually inject the decimal point based on UI setting
        dec_pos = self.config.get('dec_pos', 0)
        number = self._inject_decimal(raw_number_str, dec_pos)

        mean_conf = (sum(confs) / len(confs)) if confs else 0.0
        low = empties > 0 or (confs and min(confs) < 0.30)
        flag = "   (low-confidence)" if low else ""

        elapsed_ms = (time.time() - start_time) * 1000.0
        if number == '':
            self._last_text = "No valid number." + flag
        else:
            self._last_text = f"{number}   [{mean_conf:.2f}]{flag}"
        self.result_ready.emit(self._last_text, elapsed_ms, vis)

    def _read_digit_vote(self, binary, thick, box):
        best_txt, best_conf = '', 0.0
        for img in (binary, thick):
            res = self.reader.recognize(
                img, horizontal_list=[box], free_list=[],
                allowlist='0123456789', detail=1, paragraph=False,
                decoder='beamsearch', beamWidth=10 
            )
            if res:
                t = ''.join(c for c in str(res[0][1]) if c.isdigit())
                c = float(res[0][2])
                if t and c > best_conf:
                    best_txt, best_conf = t[0], c
        return best_txt, best_conf

    @staticmethod
    def _inject_decimal(num_str: str, dec_pos: int) -> str:
        s = re.sub(r'[^0-9]', '', num_str) # Strip to pure digits
        if not s or dec_pos == 0:
            return s
        
        # Pad with leading zeros if the number is too short (e.g., '5' -> '0.05')
        if len(s) <= dec_pos:
            s = s.zfill(dec_pos + 1)
            
        return s[:-dec_pos] + '.' + s[-dec_pos:]

    # ── Legacy CRAFT pipeline ───────────────────────
    def _run_legacy(self):
        start_time = time.time()

        binary = compute_binary(
            self.frame, self.roi,
            self.config.get('use_otsu', True),
            self.config.get('otsu_offset', 5),
            self.config.get('manual_thresh', 127),
            self.config.get('area_threshold', 0),
            self.config.get('morph_iters', 0),
            self.config.get('gap_limit', 50),
            do_splice=True,
        )

        ocr_input = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        raw = self.reader.readtext(
            ocr_input,
            allowlist='0123456789', # Removed decimal from allowlist
            width_ths=0.0,
            link_threshold=0.9,
            mag_ratio=1.5,
            text_threshold=0.1,
            low_text=0.1,
            decoder='beamsearch',
            beamWidth=10
        )

        self._last_mode   = 'legacy'
        self._last_raw    = raw
        self._last_binary = binary.copy()

        conf  = self.config.get('confidence', 0.15)
        dec_pos = self.config.get('dec_pos', 0)
        lines = self._parse_result(raw, conf, dec_pos)
        
        text  = "\n".join(lines) if lines else "No text detected."
        elapsed_ms = (time.time() - start_time) * 1000.0
        self.result_ready.emit(text, elapsed_ms, binary.copy())

    def _parse_result(self, result, confidence: float, dec_pos: int) -> list:
        if not result:
            return []

        valid_items = []
        for item in result:
            try:
                text  = str(item[1]).replace(" ", "")
                score = float(item[2])
                x_pos = float(item[0][0][0])
                x_max = max([pt[0] for pt in item[0]])

                if text and score >= confidence:
                    valid_items.append({
                        'text': text,
                        'score': score,
                        'x_min': x_pos,
                        'x_max': x_max
                    })
            except (IndexError, TypeError, ValueError):
                continue

        if not valid_items:
            return []

        valid_items.sort(key=lambda x: x['x_min'])

        grouped_results = []
        current_text = valid_items[0]['text']
        current_scores = [valid_items[0]['score']]
        last_x_max = valid_items[0]['x_max']

        for i in range(1, len(valid_items)):
            item = valid_items[i]
            distance = item['x_min'] - last_x_max

            if distance < 300:
                current_text += item['text']
                current_scores.append(item['score'])
            else:
                avg_score = sum(current_scores) / len(current_scores)
                formatted_text = self._inject_decimal(current_text, dec_pos)
                grouped_results.append(f"{formatted_text}   [{avg_score:.2f}]")
                current_text = item['text']
                current_scores = [item['score']]

            last_x_max = item['x_max']

        avg_score = sum(current_scores) / len(current_scores)
        formatted_text = self._inject_decimal(current_text, dec_pos)
        grouped_results.append(f"{formatted_text}   [{avg_score:.2f}]")

        return grouped_results


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
        self.setWindowTitle("OCR Pipeline — Biased Otsu & Iteration Healing")
        self.setGeometry(100, 100, 1260, 980)

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
        thumb_col.addWidget(QLabel("Preprocessed + segmentation (what OCR sees):"))
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

        # Recognition mode
        group_mode = QGroupBox("Recognition Mode")
        mode_layout = QVBoxLayout()
        self.chk_perdigit = QCheckBox("Per-Digit Segmentation (fixes repeated zeros)")
        self.chk_perdigit.setChecked(True)
        self.chk_perdigit.stateChanged.connect(self._refresh_preview_and_queue_ocr)
        mode_layout.addWidget(self.chk_perdigit)
        mode_hint = QLabel("ON: connected-component segmentation, one digit per inference. "
                           "OFF: legacy CRAFT path.")
        mode_hint.setStyleSheet("color: gray; font-size: 11px;")
        mode_hint.setWordWrap(True)
        mode_layout.addWidget(mode_hint)

        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("Box Padding %:"))
        self.slider_pad = QSlider(Qt.Horizontal)
        self.slider_pad.setRange(0, 60)
        self.slider_pad.setValue(30)
        self.slider_pad.setTickInterval(10)
        self.slider_pad.setTickPosition(QSlider.TicksBelow)
        self.slider_pad.valueChanged.connect(self._on_pad_changed)
        pad_row.addWidget(self.slider_pad)
        self.lbl_pad_val = QLabel("30")
        self.lbl_pad_val.setFixedWidth(30)
        pad_row.addWidget(self.lbl_pad_val)
        mode_layout.addLayout(pad_row)
        pad_hint = QLabel("Enforces a minimum aspect ratio so '7' doesn't look like '1'.")
        pad_hint.setStyleSheet("color: gray; font-size: 11px;")
        pad_hint.setWordWrap(True)
        mode_layout.addWidget(pad_hint)
        group_mode.setLayout(mode_layout)
        right_layout.addWidget(group_mode)

        # Manual Decimal Placement
        group_decimal = QGroupBox("Manual Decimal Placement")
        dec_layout = QHBoxLayout()
        dec_layout.addWidget(QLabel("Decimal Pos (from right):"))
        self.spin_dec = QSpinBox()
        self.spin_dec.setRange(0, 5)
        self.spin_dec.setValue(0)
        self.spin_dec.valueChanged.connect(self._refresh_preview_and_queue_ocr)
        dec_layout.addWidget(self.spin_dec)
        
        dec_hint = QLabel("0 = No decimal (1234)\n1 = One place (123.4)")
        dec_hint.setStyleSheet("color: gray; font-size: 11px;")
        dec_layout.addWidget(dec_hint)
        
        group_decimal.setLayout(dec_layout)
        right_layout.addWidget(group_decimal)

        # Binarization
        group_pre = QGroupBox("Binarization (Now with Biased Otsu!)")
        pre_layout = QVBoxLayout()

        otsu_row = QHBoxLayout()
        self.chk_otsu = QCheckBox("Auto Otsu Baseline")
        self.chk_otsu.setChecked(True)
        self.chk_otsu.stateChanged.connect(self._on_otsu_toggled)
        otsu_row.addWidget(self.chk_otsu)
        
        otsu_row.addWidget(QLabel("Offset:"))
        self.spin_otsu_offset = QSpinBox()
        self.spin_otsu_offset.setRange(-50, 50)
        self.spin_otsu_offset.setValue(5) # The +5 you requested!
        self.spin_otsu_offset.valueChanged.connect(self._refresh_preview_and_queue_ocr)
        otsu_row.addWidget(self.spin_otsu_offset)
        otsu_row.addStretch()
        pre_layout.addLayout(otsu_row)
        
        otsu_hint = QLabel("Otsu calculates the ideal math, the Offset bumps it up (+5) to capture glared LCD segments.")
        otsu_hint.setStyleSheet("color: #0055cc; font-size: 11px;")
        pre_layout.addWidget(otsu_hint)

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

        # Morphology & Area Filter
        group_filters = QGroupBox("Morphology & Noise Filter")
        filter_layout = QVBoxLayout()

        # Area Filter Slider
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("1. Min Area (Kill dots):"))
        self.slider_area = QSlider(Qt.Horizontal)
        self.slider_area.setRange(0, 200)
        self.slider_area.setValue(80)
        self.slider_area.setTickInterval(20)
        self.slider_area.setTickPosition(QSlider.TicksBelow)
        self.slider_area.valueChanged.connect(self._on_area_changed)
        area_row.addWidget(self.slider_area)
        self.lbl_area_val = QLabel("80")
        self.lbl_area_val.setFixedWidth(30)
        area_row.addWidget(self.lbl_area_val)
        filter_layout.addLayout(area_row)
        
        area_hint = QLabel("Deletes isolated dots FIRST before they get glued to numbers.")
        area_hint.setStyleSheet("color: gray; font-size: 11px;")
        filter_layout.addWidget(area_hint)

        # Closing Slider (REVERTED TO ITERATIONS)
        morph_row = QHBoxLayout()
        morph_row.addWidget(QLabel("2. Closing (Heal gaps):"))
        self.slider_morph = QSlider(Qt.Horizontal)
        self.slider_morph.setRange(0, 5)
        self.slider_morph.setValue(1)
        self.slider_morph.setTickInterval(1)
        self.slider_morph.setTickPosition(QSlider.TicksBelow)
        self.slider_morph.valueChanged.connect(self._on_morph_changed)
        morph_row.addWidget(self.slider_morph)
        self.lbl_morph_val = QLabel("1")
        self.lbl_morph_val.setFixedWidth(30)
        morph_row.addWidget(self.lbl_morph_val)
        filter_layout.addLayout(morph_row)
        
        morph_hint = QLabel("Number of times to run the healing brush to melt broken segments back together.")
        morph_hint.setStyleSheet("color: gray; font-size: 11px;")
        filter_layout.addWidget(morph_hint)

        group_filters.setLayout(filter_layout)
        right_layout.addWidget(group_filters)

        # Confidence Filter
        group_conf = QGroupBox("OCR Confidence Filter (legacy path)")
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

        conf_hint = QLabel("Per-Digit mode never drops digits, so this is a display value there.")
        conf_hint.setStyleSheet("color: gray; font-size: 11px;")
        conf_layout.addWidget(conf_hint)

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
        self.spin_otsu_offset.setEnabled(bool(state))
        self._refresh_preview_and_queue_ocr()

    def _on_thresh_changed(self, value):
        self.lbl_thresh_val.setText(str(value))
        self._refresh_preview_and_queue_ocr()

    def _on_morph_changed(self, value):
        self.lbl_morph_val.setText(str(value))
        self._refresh_preview_and_queue_ocr()

    def _on_area_changed(self, value):
        self.lbl_area_val.setText(str(value))
        self._refresh_preview_and_queue_ocr()

    def _on_gap_changed(self, value):
        self.lbl_gap_val.setText(str(value))
        self._refresh_preview_and_queue_ocr()

    def _on_pad_changed(self, value):
        self.lbl_pad_val.setText(str(value))
        self._refresh_preview_and_queue_ocr()

    def _refresh_preview_and_queue_ocr(self):
        if self.current_frame is None:
            return
        per_digit = self.chk_perdigit.isChecked()
        binary = compute_binary(
            self.current_frame,
            self.selected_roi,
            self.chk_otsu.isChecked(),
            self.spin_otsu_offset.value(),
            self.slider_thresh.value(),
            self.slider_area.value(),
            self.slider_morph.value(),
            0, # gap limit passed as 0 since we removed the slider for it to make room
            do_splice=not per_digit,
        )
        if per_digit:
            dboxes = segment_digits(binary)
            ph, pw = binary.shape
            p = self.slider_pad.value() / 100.0
            pboxes = pad_digit_boxes(dboxes, pw, ph, pad_x=p, pad_y=p)
            self._show_preview(annotate_segmentation(binary, pboxes))
        else:
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
                'per_digit':      self.chk_perdigit.isChecked(),
                'pad_pct':        self.slider_pad.value(),
                'use_otsu':       self.chk_otsu.isChecked(),
                'otsu_offset':    self.spin_otsu_offset.value(),
                'manual_thresh':  self.slider_thresh.value(),
                'area_threshold': self.slider_area.value(),
                'morph_iters':    self.slider_morph.value(),
                'gap_limit':      50, # hardcoded fallback for legacy
                'confidence':     self.conf_slider.value() / 100.0,
                'dec_pos':        self.spin_dec.value(),
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

    def _show_preview(self, img: np.ndarray):
        if img.ndim == 2:
            h, w = img.shape
            q_img = QImage(bytes(img.data), w, h, w, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(bytes(rgb.data), w, h, ch * w, QImage.Format_RGB888)
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
