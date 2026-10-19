import os
from typing import Dict, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QTabWidget, QWidget, QMessageBox, QRadioButton, QButtonGroup,
    QCheckBox
)
from PyQt6.QtCore import Qt

class SettingsDialog(QDialog):
    """Hộp thoại cài đặt tham số mô hình Whisper, OCR phụ đề và ngưỡng phân loại."""

    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Cài Đặt Tham Số & Thuật Toán - VoxSort AI")
        self.setMinimumWidth(580)
        self.config = config.copy()

        self._init_ui()
        self._load_values()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)

        tabs = QTabWidget()

        # TAB 1: Ngưỡng Phân Loại
        tab_thresholds = QWidget()
        thresh_layout = QVBoxLayout(tab_thresholds)
        thresh_layout.setSpacing(14)

        # Nhóm Unbox
        unbox_group = QGroupBox("📦 Tiêu chuẩn phân loại: UNBOX (Không Voice / Nhạc nền)")
        unbox_grid = QGridLayout(unbox_group)
        unbox_grid.setVerticalSpacing(10)

        unbox_grid.addWidget(QLabel("Tổng số từ tối đa:"), 0, 0)
        self.spin_unbox_words = QSpinBox()
        self.spin_unbox_words.setRange(0, 100)
        self.spin_unbox_words.setToolTip("Nếu tổng số từ nói nhỏ hơn giá trị này -> Xếp vào Unbox")
        unbox_grid.addWidget(self.spin_unbox_words, 0, 1)

        unbox_grid.addWidget(QLabel("Thời lượng nói tối đa (giây):"), 1, 0)
        self.spin_unbox_dur = QDoubleSpinBox()
        self.spin_unbox_dur.setRange(0.0, 60.0)
        self.spin_unbox_dur.setSingleStep(0.5)
        self.spin_unbox_dur.setToolTip("Nếu thời lượng nói ngắt quãng nhỏ hơn giá trị này -> Xếp vào Unbox")
        unbox_grid.addWidget(self.spin_unbox_dur, 1, 1)

        thresh_layout.addWidget(unbox_group)

        # Nhóm Review
        review_group = QGroupBox("🎙️ Tiêu chuẩn phân loại: REVIEW (Có Voice Thuyết Minh)")
        review_grid = QGridLayout(review_group)
        review_grid.setVerticalSpacing(10)

        review_grid.addWidget(QLabel("Tổng số từ tối thiểu:"), 0, 0)
        self.spin_review_words = QSpinBox()
        self.spin_review_words.setRange(1, 1000)
        self.spin_review_words.setToolTip("Nếu số từ >= giá trị này VÀ tốc độ nói đạt chuẩn -> Xếp vào Review")
        review_grid.addWidget(self.spin_review_words, 0, 1)

        review_grid.addWidget(QLabel("Tốc độ nói tối thiểu (từ / giây):"), 1, 0)
        self.spin_review_speed = QDoubleSpinBox()
        self.spin_review_speed.setRange(0.1, 10.0)
        self.spin_review_speed.setSingleStep(0.1)
        self.spin_review_speed.setToolTip("Tốc độ nói trung bình (từ / tổng thời gian nói)")
        review_grid.addWidget(self.spin_review_speed, 1, 1)

        thresh_layout.addWidget(review_group)

        note_label = QLabel("<i>* Lưu ý: Các video không thỏa mãn cả 2 điều kiện trên sẽ tự động được xếp vào nhóm <b>Cần kiểm tra lại</b>.</i>")
        note_label.setStyleSheet("color: #94A3B8; font-size: 11px;")
        thresh_layout.addWidget(note_label)
        thresh_layout.addStretch()

        tabs.addTab(tab_thresholds, "🎯 Ngưỡng Phân Loại")

        # TAB 2: Cấu hình Whisper & Phụ Đề OCR
        tab_ai = QWidget()
        ai_layout = QVBoxLayout(tab_ai)
        ai_layout.setSpacing(14)

        ai_group = QGroupBox("🤖 Cấu hình Faster-Whisper Model")
        ai_grid = QGridLayout(ai_group)
        ai_grid.setVerticalSpacing(10)

        ai_grid.addWidget(QLabel("Kích thước Model Whisper:"), 0, 0)
        self.combo_model = QComboBox()
        self.combo_model.addItems(["tiny", "base", "small", "medium"])
        self.combo_model.setToolTip("tiny: Nhanh nhất (khuyên dùng)\nbase: Cân bằng\nsmall: Chính xác cao hơn")
        ai_grid.addWidget(self.combo_model, 0, 1)

        ai_grid.addWidget(QLabel("Thiết bị xử lý (Device):"), 1, 0)
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cpu", "cuda"])
        ai_grid.addWidget(self.combo_device, 1, 1)

        ai_grid.addWidget(QLabel("Kiểu tính toán (Compute Type):"), 2, 0)
        self.combo_compute = QComboBox()
        self.combo_compute.addItems(["int8", "float32", "float16"])
        ai_grid.addWidget(self.combo_compute, 2, 1)

        ai_grid.addWidget(QLabel("Ngưỡng lọc im lặng (VAD No-Speech):"), 3, 0)
        self.spin_vad = QDoubleSpinBox()
        self.spin_vad.setRange(0.1, 0.99)
        self.spin_vad.setSingleStep(0.05)
        self.spin_vad.setToolTip("Mặc định 0.6. Bỏ qua các đoạn âm thanh nếu xác suất không phải giọng nói > 0.6")
        ai_grid.addWidget(self.spin_vad, 3, 1)

        ai_grid.addWidget(QLabel("Giới hạn quét âm thanh (giây):"), 4, 0)
        self.spin_scan_limit = QSpinBox()
        self.spin_scan_limit.setRange(0, 3600)
        self.spin_scan_limit.setSingleStep(30)
        self.spin_scan_limit.setSpecialValueText("0 (Quét toàn bộ video)")
        self.spin_scan_limit.setToolTip("0 = Quét hết video. Đặt ví dụ 60 hoặc 120 để chỉ quét N giây đầu, giúp tăng tốc độ gấp nhiều lần!")
        ai_grid.addWidget(self.spin_scan_limit, 4, 1)

        ai_layout.addWidget(ai_group)
        ai_layout.addStretch()
        tabs.addTab(tab_ai, "⚡ Mô hình Whisper Voice")

        # TAB 3: Thư mục đích & Hành động
        tab_folders = QWidget()
        fold_layout = QVBoxLayout(tab_folders)
        fold_layout.setSpacing(14)

        fold_group = QGroupBox("📁 Tên thư mục phân loại tự động")
        fold_grid = QGridLayout(fold_group)
        fold_grid.setVerticalSpacing(10)

        fold_grid.addWidget(QLabel("Thư mục Unbox (Không Voice):"), 0, 0)
        self.edit_unbox_folder = QLineEdit()
        fold_grid.addWidget(self.edit_unbox_folder, 0, 1)

        fold_grid.addWidget(QLabel("  ↳ Thư mục con Có Nhạc:"), 1, 0)
        self.edit_unbox_music = QLineEdit()
        fold_grid.addWidget(self.edit_unbox_music, 1, 1)

        fold_grid.addWidget(QLabel("  ↳ Thư mục con Không Nhạc:"), 2, 0)
        self.edit_unbox_nomusic = QLineEdit()
        fold_grid.addWidget(self.edit_unbox_nomusic, 2, 1)

        fold_grid.addWidget(QLabel("Thư mục Review (Có Voice):"), 3, 0)
        self.edit_review_folder = QLineEdit()
        fold_grid.addWidget(self.edit_review_folder, 3, 1)

        fold_grid.addWidget(QLabel("Thư mục Nghi ngờ / Cần KT:"), 4, 0)
        self.edit_doubt_folder = QLineEdit()
        fold_grid.addWidget(self.edit_doubt_folder, 4, 1)

        fold_layout.addWidget(fold_group)

        # Chế độ Di chuyển / Copy
        act_group = QGroupBox("🔄 Hành động với file video")
        act_layout = QHBoxLayout(act_group)
        self.radio_move = QRadioButton("Di chuyển file (Move - Tiết kiệm dung lượng)")
        self.radio_copy = QRadioButton("Sao chép file (Copy - Giữ nguyên file gốc an toàn)")
        self.act_group = QButtonGroup()
        self.act_group.addButton(self.radio_move)
        self.act_group.addButton(self.radio_copy)
        act_layout.addWidget(self.radio_move)
        act_layout.addWidget(self.radio_copy)
        fold_layout.addWidget(act_group)

        fold_layout.addStretch()
        tabs.addTab(tab_folders, "📂 Thư Mục & Tác Vụ")

        # TAB 4: Bộ Lọc Tự Động Xóa (Chuẩn Shopee)
        tab_filters = QWidget()
        filter_layout = QVBoxLayout(tab_filters)
        filter_layout.setSpacing(14)

        filt_group = QGroupBox("🧹 Tiêu chuẩn tự động XÓA video kém chất lượng (Chuẩn Shopee Video)")
        filt_grid = QGridLayout(filt_group)
        filt_grid.setVerticalSpacing(12)
        filt_grid.setHorizontalSpacing(10)

        # 1. Độ phân giải < 720p
        self.chk_del_low_res = QCheckBox("Tự động XÓA video có độ phân giải dưới 720p")
        self.chk_del_low_res.setStyleSheet("font-weight: 600; color: #FCA5A5;")
        filt_grid.addWidget(self.chk_del_low_res, 0, 0, 1, 2)

        filt_grid.addWidget(QLabel("Ngưỡng phân giải tối thiểu:"), 1, 0)
        self.spin_min_res = QSpinBox()
        self.spin_min_res.setRange(240, 2160)
        self.spin_min_res.setSingleStep(120)
        self.spin_min_res.setSuffix("p (cạnh ngắn)")
        self.spin_min_res.setValue(720)
        self.spin_min_res.setToolTip("Video có cạnh ngắn < giá trị này sẽ bị xóa (ví dụ: < 720p như 360p, 480p, 540p)")
        filt_grid.addWidget(self.spin_min_res, 1, 1)

        # 2. Định dạng không phải 9:16
        self.chk_del_non_9_16 = QCheckBox("Tự động XÓA video có định dạng không phải 9:16")
        self.chk_del_non_9_16.setStyleSheet("font-weight: 600; color: #FCA5A5;")
        self.chk_del_non_9_16.setToolTip("Chỉ giữ lại video dọc chuẩn 9:16 (Shopee/TikTok). Video ngang 16:9, vuông 1:1, 3:4,... sẽ bị xóa.")
        filt_grid.addWidget(self.chk_del_non_9_16, 2, 0, 1, 2)

        lbl_ratio_note = QLabel("<i>(Chỉ giữ lại video dọc chuẩn tỉ lệ 9:16 như 1080x1920, 720x1280. Xóa các video ngang 16:9, vuông 1:1,...)</i>")
        lbl_ratio_note.setStyleSheet("color: #94A3B8; font-size: 11px;")
        filt_grid.addWidget(lbl_ratio_note, 3, 0, 1, 2)

        # 3. Thời lượng < 10s
        self.chk_del_short = QCheckBox("Tự động XÓA video có thời lượng dưới 10 giây")
        self.chk_del_short.setStyleSheet("font-weight: 600; color: #FCA5A5;")
        filt_grid.addWidget(self.chk_del_short, 4, 0, 1, 2)

        filt_grid.addWidget(QLabel("Thời lượng tối thiểu:"), 5, 0)
        self.spin_min_dur = QDoubleSpinBox()
        self.spin_min_dur.setRange(1.0, 300.0)
        self.spin_min_dur.setSingleStep(1.0)
        self.spin_min_dur.setSuffix(" giây")
        self.spin_min_dur.setValue(10.0)
        self.spin_min_dur.setToolTip("Video có thời lượng nhỏ hơn số giây này sẽ tự động bị xóa.")
        filt_grid.addWidget(self.spin_min_dur, 5, 1)

        filter_layout.addWidget(filt_group)

        warn_box = QLabel(
            "<b>⚠️ Lưu ý an toàn & Tối ưu hiệu năng:</b><br>"
            "Các video vi phạm tiêu chuẩn trên sẽ <b>bị xóa trực tiếp khỏi thư mục nguồn</b> ngay khi bắt đầu quét.<br>"
            "Cơ chế này giúp bỏ qua phân tích Whisper AI cho video rác, tăng tốc độ xử lý lên nhiều lần!"
        )
        warn_box.setStyleSheet("background-color: #2D1515; color: #FCA5A5; border: 1px solid #7F1D1D; border-radius: 8px; padding: 10px; font-size: 12px;")
        filter_layout.addWidget(warn_box)
        filter_layout.addStretch()

        tabs.addTab(tab_filters, "🧹 Bộ Lọc Tự Động Xóa")

        main_layout.addWidget(tabs)

        # Buttons footer
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("Khôi phục mặc định")
        btn_reset.setStyleSheet("background-color: #334155;")
        btn_reset.clicked.connect(self._reset_defaults)

        btn_save = QPushButton("Lưu cài đặt")
        btn_save.setStyleSheet("background-color: #6366F1; font-weight: bold; color: white;")
        btn_save.clicked.connect(self._save_and_close)

        btn_cancel = QPushButton("Hủy bỏ")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)

        main_layout.addLayout(btn_layout)

    def _load_values(self):
        c = self.config
        self.spin_unbox_words.setValue(c.get("unbox_max_words", 8))
        self.spin_unbox_dur.setValue(c.get("unbox_max_duration", 2.0))
        self.spin_review_words.setValue(c.get("review_min_words", 15))
        self.spin_review_speed.setValue(c.get("review_min_speed", 1.5))

        model_name = c.get("model_name", "tiny")
        idx = self.combo_model.findText(model_name)
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)

        device = c.get("device", "cpu")
        idx_dev = self.combo_device.findText(device)
        if idx_dev >= 0:
            self.combo_device.setCurrentIndex(idx_dev)

        compute = c.get("compute_type", "int8")
        idx_comp = self.combo_compute.findText(compute)
        if idx_comp >= 0:
            self.combo_compute.setCurrentIndex(idx_comp)

        self.spin_vad.setValue(c.get("no_speech_prob", 0.6))
        self.spin_scan_limit.setValue(c.get("scan_duration_limit", 0))

        self.edit_unbox_folder.setText(c.get("output_unbox", "1_Unbox_Khong_Voice"))
        self.edit_unbox_music.setText(c.get("output_unbox_music", "1_Co_Nhac"))
        self.edit_unbox_nomusic.setText(c.get("output_unbox_nomusic", "2_Khong_Nhac"))
        self.edit_review_folder.setText(c.get("output_review", "2_Review_Co_Voice"))
        self.edit_doubt_folder.setText(c.get("output_doubt", "3_Can_Kiem_Tra_Lai"))

        if c.get("action_mode", "move") == "copy":
            self.radio_copy.setChecked(True)
        else:
            self.radio_move.setChecked(True)

        # Bộ lọc Shopee
        self.chk_del_low_res.setChecked(c.get("delete_low_res", True))
        self.spin_min_res.setValue(c.get("filter_min_resolution", 720))
        self.chk_del_non_9_16.setChecked(c.get("delete_non_9_16", True))
        self.chk_del_short.setChecked(c.get("delete_short_duration", True))
        self.spin_min_dur.setValue(c.get("filter_min_duration", 10.0))

    def _reset_defaults(self):
        self.spin_unbox_words.setValue(8)
        self.spin_unbox_dur.setValue(2.0)
        self.spin_review_words.setValue(15)
        self.spin_review_speed.setValue(1.5)
        self.combo_model.setCurrentText("tiny")
        self.combo_device.setCurrentText("cpu")
        self.combo_compute.setCurrentText("int8")
        self.spin_vad.setValue(0.6)
        self.spin_scan_limit.setValue(0)
        self.edit_unbox_folder.setText("1_Unbox_Khong_Voice")
        self.edit_unbox_music.setText("1_Co_Nhac")
        self.edit_unbox_nomusic.setText("2_Khong_Nhac")
        self.edit_review_folder.setText("2_Review_Co_Voice")
        self.edit_doubt_folder.setText("3_Can_Kiem_Tra_Lai")
        self.radio_move.setChecked(True)

        # Mặc định bộ lọc Shopee
        self.chk_del_low_res.setChecked(True)
        self.spin_min_res.setValue(720)
        self.chk_del_non_9_16.setChecked(True)
        self.chk_del_short.setChecked(True)
        self.spin_min_dur.setValue(10.0)

    def _save_and_close(self):
        self.config["unbox_max_words"] = self.spin_unbox_words.value()
        self.config["unbox_max_duration"] = self.spin_unbox_dur.value()
        self.config["review_min_words"] = self.spin_review_words.value()
        self.config["review_min_speed"] = self.spin_review_speed.value()

        self.config["model_name"] = self.combo_model.currentText()
        self.config["device"] = self.combo_device.currentText()
        self.config["compute_type"] = self.combo_compute.currentText()
        self.config["no_speech_prob"] = round(self.spin_vad.value(), 2)
        self.config["scan_duration_limit"] = self.spin_scan_limit.value()

        self.config["split_review_subtitles"] = False

        self.config["output_unbox"] = self.edit_unbox_folder.text().strip() or "1_Unbox_Khong_Voice"
        self.config["output_unbox_music"] = self.edit_unbox_music.text().strip() or "1_Co_Nhac"
        self.config["output_unbox_nomusic"] = self.edit_unbox_nomusic.text().strip() or "2_Khong_Nhac"
        self.config["output_review"] = self.edit_review_folder.text().strip() or "2_Review_Co_Voice"
        self.config["output_doubt"] = self.edit_doubt_folder.text().strip() or "3_Can_Kiem_Tra_Lai"

        self.config["action_mode"] = "copy" if self.radio_copy.isChecked() else "move"

        # Lưu bộ lọc Shopee
        self.config["delete_low_res"] = self.chk_del_low_res.isChecked()
        self.config["filter_min_resolution"] = self.spin_min_res.value()
        self.config["delete_non_9_16"] = self.chk_del_non_9_16.isChecked()
        self.config["delete_short_duration"] = self.chk_del_short.isChecked()
        self.config["filter_min_duration"] = self.spin_min_dur.value()

        self.accept()

    def get_config(self) -> Dict[str, Any]:
        return self.config
