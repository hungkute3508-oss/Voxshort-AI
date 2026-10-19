import os
import sys
import csv
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QProgressBar, QGroupBox, QFileDialog, QMessageBox, QTabWidget,
    QPlainTextEdit, QHeaderView, QAbstractItemView, QComboBox,
    QSpinBox, QCheckBox, QRadioButton, QButtonGroup, QFrame
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor, QFont, QDesktopServices

from config_manager import load_config, save_config
from classifier_core import (
    VideoInfo, ClassificationResult, scan_video_files,
    CATEGORY_UNBOX, CATEGORY_UNBOX_MUSIC, CATEGORY_UNBOX_NOMUSIC,
    CATEGORY_REVIEW, CATEGORY_REVIEW_SUB, CATEGORY_REVIEW_NOSUB,
    CATEGORY_DOUBT, CATEGORY_ERROR, CATEGORY_DELETED, CATEGORY_NAMES
)
from gui.worker_thread import ClassifierWorker
from gui.settings_dialog import SettingsDialog
from gui.styles import DARK_THEME_QSS

class MainWindow(QMainWindow):
    """Giao diện chính VoxSort AI Desktop."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxSort AI - Phân Loại Video Tự Động (Whisper Voice AI)")
        self.resize(1240, 800)
        self.setMinimumSize(1000, 680)

        # Config & State
        self.config = load_config()
        self.scanned_videos: List[VideoInfo] = []
        self.results_map: Dict[str, ClassificationResult] = {}
        self.worker: Optional[ClassifierWorker] = None

        # Statistics
        self.count_total = 0
        self.count_unbox = 0
        self.count_unbox_music = 0
        self.count_unbox_nomusic = 0
        self.count_review = 0
        self.count_deleted = 0
        self.count_doubt = 0
        self.count_error = 0

        self.setStyleSheet(DARK_THEME_QSS)
        self._init_ui()

        # Tải lại đường dẫn gần nhất nếu có
        if self.config.get("source_dir") and os.path.isdir(self.config.get("source_dir")):
            self.edit_source.setText(self.config["source_dir"])
            self._scan_source_folder(silent=True)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(16, 14, 16, 14)
        root_layout.setSpacing(12)

        # 1. HEADER BAR
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(4, 0, 4, 0)

        title_box = QVBoxLayout()
        lbl_app = QLabel("⚡ VOXSORT AI")
        lbl_app.setStyleSheet("font-size: 20px; font-weight: 800; color: #818CF8; letter-spacing: 1px;")
        lbl_desc = QLabel("Phân loại video Unbox & Review tự động bằng Whisper Voice AI")
        lbl_desc.setStyleSheet("color: #94A3B8; font-size: 12px;")
        title_box.addWidget(lbl_app)
        title_box.addWidget(lbl_desc)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        # Nút Cài đặt
        btn_settings = QPushButton("⚙️ Cài đặt thuật toán & Ngưỡng")
        btn_settings.setStyleSheet("background-color: #1E293B; border: 1px solid #475569; padding: 7px 14px; font-weight: 600;")
        btn_settings.clicked.connect(self._open_settings)
        header_layout.addWidget(btn_settings)

        root_layout.addWidget(header_frame)

        # 2. SOURCE FOLDER & DESTINATION PANEL
        folder_box = QGroupBox("📁 Thư Mục Kho Video & Cấu Hình Đích")
        folder_layout = QVBoxLayout(folder_box)
        folder_layout.setSpacing(10)

        # Hàng 1: Chọn thư mục gốc
        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Thư mục chứa video:"))
        self.edit_source = QLineEdit()
        self.edit_source.setPlaceholderText("Chọn hoặc dán đường dẫn thư mục kho video...")
        self.edit_source.textChanged.connect(self._on_source_text_changed)
        row_src.addWidget(self.edit_source, stretch=1)

        btn_browse = QPushButton("📂 Chọn thư mục...")
        btn_browse.setStyleSheet("background-color: #4F46E5; color: white;")
        btn_browse.clicked.connect(self._browse_source_folder)
        row_src.addWidget(btn_browse)

        btn_open_src = QPushButton("↗ Mở")
        btn_open_src.setToolTip("Mở thư mục nguồn trong File Explorer")
        btn_open_src.clicked.connect(self._open_source_in_explorer)
        row_src.addWidget(btn_open_src)

        folder_layout.addLayout(row_src)

        # Hàng 2: Tùy chọn Quét con & Hành động file & Số luồng
        row_opts = QHBoxLayout()
        self.chk_recursive = QCheckBox("Quét cả thư mục con (Recursive)")
        self.chk_recursive.setChecked(self.config.get("recursive", False))
        self.chk_recursive.stateChanged.connect(self._on_option_changed)
        row_opts.addWidget(self.chk_recursive)

        row_opts.addSpacing(15)
        row_opts.addWidget(QLabel("Hành động:"))
        self.radio_move = QRadioButton("Di chuyển (Move)")
        self.radio_copy = QRadioButton("Sao chép (Copy)")
        if self.config.get("action_mode", "move") == "copy":
            self.radio_copy.setChecked(True)
        else:
            self.radio_move.setChecked(True)
        self.radio_move.toggled.connect(self._on_option_changed)
        row_opts.addWidget(self.radio_move)
        row_opts.addWidget(self.radio_copy)

        row_opts.addSpacing(15)
        row_opts.addWidget(QLabel("Số luồng:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 16)
        self.spin_workers.setValue(self.config.get("workers", 2))
        self.spin_workers.setToolTip("Số luồng CPU xử lý song song")
        self.spin_workers.valueChanged.connect(self._on_option_changed)
        row_opts.addWidget(self.spin_workers)

        row_opts.addSpacing(15)
        row_opts.addWidget(QLabel("Model:"))
        self.combo_quick_model = QComboBox()
        self.combo_quick_model.addItems(["tiny", "base", "small", "medium"])
        self.combo_quick_model.setCurrentText(self.config.get("model_name", "tiny"))
        self.combo_quick_model.currentTextChanged.connect(self._on_model_changed)
        row_opts.addWidget(self.combo_quick_model)

        row_opts.addStretch()

        btn_rescan = QPushButton("🔄 Quét lại")
        btn_rescan.setToolTip("Quét lại danh sách video trong thư mục")
        btn_rescan.clicked.connect(lambda: self._scan_source_folder(silent=False))
        row_opts.addWidget(btn_rescan)

        folder_layout.addLayout(row_opts)

        # Hàng 2.5: Bộ lọc tự động xóa mặc định (Chuẩn Shopee Video)
        row_filters = QHBoxLayout()
        lbl_flt_title = QLabel("🧹 Bộ lọc tự động xóa:")
        lbl_flt_title.setStyleSheet("font-weight: 700; color: #FCA5A5;")
        row_filters.addWidget(lbl_flt_title)

        self.chk_del_low_res = QCheckBox("Xóa < 720p")
        self.chk_del_low_res.setChecked(self.config.get("delete_low_res", True))
        self.chk_del_low_res.setToolTip("Tự động xóa video có độ phân giải dưới 720p (cạnh ngắn < 720px)")
        self.chk_del_low_res.stateChanged.connect(self._on_option_changed)
        row_filters.addWidget(self.chk_del_low_res)

        self.chk_del_non_9_16 = QCheckBox("Xóa không phải 9:16")
        self.chk_del_non_9_16.setChecked(self.config.get("delete_non_9_16", True))
        self.chk_del_non_9_16.setToolTip("Tự động xóa video không đúng tỉ lệ dọc chuẩn 9:16 (Shopee/TikTok)")
        self.chk_del_non_9_16.stateChanged.connect(self._on_option_changed)
        row_filters.addWidget(self.chk_del_non_9_16)

        self.chk_del_short = QCheckBox("Xóa < 10s")
        self.chk_del_short.setChecked(self.config.get("delete_short_duration", True))
        self.chk_del_short.setToolTip("Tự động xóa video có thời lượng dưới 10 giây")
        self.chk_del_short.stateChanged.connect(self._on_option_changed)
        row_filters.addWidget(self.chk_del_short)

        lbl_filter_hint = QLabel("(Thiết lập mặc định - Tự động xóa rác, bỏ qua Whisper để tăng tốc)")
        lbl_filter_hint.setStyleSheet("color: #64748B; font-size: 11px;")
        row_filters.addWidget(lbl_filter_hint)

        row_filters.addStretch()
        folder_layout.addLayout(row_filters)

        # Hàng 3: Các nút mở nhanh thư mục kết quả
        row_targets = QHBoxLayout()
        row_targets.addWidget(QLabel("Mở thư mục kết quả:"))

        self.btn_open_review = QPushButton("🎙️ Mở: Review")
        self.btn_open_review.setStyleSheet("color: #34D399; border-color: #047857;")
        self.btn_open_review.clicked.connect(lambda: self._open_dest_folder(CATEGORY_REVIEW))
        row_targets.addWidget(self.btn_open_review)

        self.btn_open_unbox = QPushButton("📦 Mở: Unbox")
        self.btn_open_unbox.setStyleSheet("color: #38BDF8; border-color: #0369A1;")
        self.btn_open_unbox.clicked.connect(lambda: self._open_dest_folder(CATEGORY_UNBOX))
        row_targets.addWidget(self.btn_open_unbox)

        self.btn_open_unbox_music = QPushButton("🎵 Có Nhạc")
        self.btn_open_unbox_music.setStyleSheet("color: #38BDF8; border-color: #0284C7; font-size: 11px;")
        self.btn_open_unbox_music.clicked.connect(lambda: self._open_dest_folder(CATEGORY_UNBOX_MUSIC))
        row_targets.addWidget(self.btn_open_unbox_music)

        self.btn_open_unbox_nomusic = QPushButton("🔇 Không Nhạc")
        self.btn_open_unbox_nomusic.setStyleSheet("color: #94A3B8; border-color: #475569; font-size: 11px;")
        self.btn_open_unbox_nomusic.clicked.connect(lambda: self._open_dest_folder(CATEGORY_UNBOX_NOMUSIC))
        row_targets.addWidget(self.btn_open_unbox_nomusic)

        self.btn_open_doubt = QPushButton("⚠️ Mở: Cần KT")
        self.btn_open_doubt.setStyleSheet("color: #FBBF24; border-color: #B45309;")
        self.btn_open_doubt.clicked.connect(lambda: self._open_dest_folder(CATEGORY_DOUBT))
        row_targets.addWidget(self.btn_open_doubt)

        row_targets.addStretch()
        folder_layout.addLayout(row_targets)

        root_layout.addWidget(folder_box)

        # 3. METRIC DASHBOARD CARDS
        cards_frame = QFrame()
        cards_layout = QHBoxLayout(cards_frame)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(10)

        self.card_total = self._create_metric_card("TỔNG VIDEO QUÉT", "0", "#94A3B8")
        self.card_review = self._create_metric_card("REVIEW (CÓ VOICE)", "0", "#34D399")
        self.card_unbox = self._create_metric_card("UNBOX (KHÔNG VOICE)", "0", "#38BDF8")
        self.card_deleted = self._create_metric_card("ĐÃ XÓA (KÉM CHUẨN)", "0", "#EF4444")
        self.card_doubt = self._create_metric_card("CẦN KIỂM TRA LẠI", "0", "#FBBF24")
        self.card_error = self._create_metric_card("LỖI XỬ LÝ", "0", "#F87171")

        cards_layout.addWidget(self.card_total)
        cards_layout.addWidget(self.card_review)
        cards_layout.addWidget(self.card_unbox)
        cards_layout.addWidget(self.card_deleted)
        cards_layout.addWidget(self.card_doubt)
        cards_layout.addWidget(self.card_error)

        root_layout.addWidget(cards_frame)

        # 4. ACTION & PROGRESS CONTROLS
        ctrl_frame = QFrame()
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(6)

        # Progress bar
        self.prog_bar = QProgressBar()
        self.prog_bar.setValue(0)
        self.prog_bar.setFormat("Chờ bắt đầu...")
        ctrl_layout.addWidget(self.prog_bar)

        # Controls & status text
        row_ctrl = QHBoxLayout()

        self.btn_start = QPushButton("🚀 BẮT ĐẦU PHÂN LOẠI")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self._start_classification)
        row_ctrl.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸️ Tạm dừng")
        self.btn_pause.setObjectName("btnPause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        row_ctrl.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹️ Dừng quét")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_classification)
        row_ctrl.addWidget(self.btn_stop)

        self.lbl_progress_info = QLabel("Sẵn sàng quét video.")
        self.lbl_progress_info.setStyleSheet("color: #94A3B8; font-size: 12px; margin-left: 10px;")
        row_ctrl.addWidget(self.lbl_progress_info, stretch=1)

        btn_export = QPushButton("📥 Xuất Báo Cáo CSV")
        btn_export.setStyleSheet("background-color: #1E293B; border: 1px solid #475569;")
        btn_export.clicked.connect(self._export_to_csv)
        row_ctrl.addWidget(btn_export)

        ctrl_layout.addLayout(row_ctrl)
        root_layout.addWidget(ctrl_frame)

        # 5. MAIN CONTENT TABS (TABLE & LOGS)
        self.tabs = QTabWidget()

        # TAB 1: Live Results Table
        tab_table = QWidget()
        tab_table_layout = QVBoxLayout(tab_table)
        tab_table_layout.setContentsMargins(8, 8, 8, 8)
        tab_table_layout.setSpacing(8)

        # Search and filters
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Bộ lọc:"))

        self.btn_filter_all = QPushButton("Tất cả (0)")
        self.btn_filter_all.setCheckable(True)
        self.btn_filter_all.setChecked(True)
        self.btn_filter_all.clicked.connect(lambda: self._apply_filter("ALL"))
        filter_row.addWidget(self.btn_filter_all)

        self.btn_filter_review = QPushButton("🎙️ Review (0)")
        self.btn_filter_review.setCheckable(True)
        self.btn_filter_review.clicked.connect(lambda: self._apply_filter(CATEGORY_REVIEW))
        filter_row.addWidget(self.btn_filter_review)

        self.btn_filter_unbox = QPushButton("📦 Unbox (0)")
        self.btn_filter_unbox.setCheckable(True)
        self.btn_filter_unbox.clicked.connect(lambda: self._apply_filter(CATEGORY_UNBOX))
        filter_row.addWidget(self.btn_filter_unbox)

        self.btn_filter_unbox_music = QPushButton("🎵 Có Nhạc (0)")
        self.btn_filter_unbox_music.setCheckable(True)
        self.btn_filter_unbox_music.clicked.connect(lambda: self._apply_filter(CATEGORY_UNBOX_MUSIC))
        filter_row.addWidget(self.btn_filter_unbox_music)

        self.btn_filter_unbox_nomusic = QPushButton("🔇 Không Nhạc (0)")
        self.btn_filter_unbox_nomusic.setCheckable(True)
        self.btn_filter_unbox_nomusic.clicked.connect(lambda: self._apply_filter(CATEGORY_UNBOX_NOMUSIC))
        filter_row.addWidget(self.btn_filter_unbox_nomusic)

        self.btn_filter_deleted = QPushButton("🗑️ Đã xóa (0)")
        self.btn_filter_deleted.setCheckable(True)
        self.btn_filter_deleted.clicked.connect(lambda: self._apply_filter(CATEGORY_DELETED))
        filter_row.addWidget(self.btn_filter_deleted)

        self.btn_filter_doubt = QPushButton("⚠️ Cần KT (0)")
        self.btn_filter_doubt.setCheckable(True)
        self.btn_filter_doubt.clicked.connect(lambda: self._apply_filter(CATEGORY_DOUBT))
        filter_row.addWidget(self.btn_filter_doubt)

        self.btn_filter_error = QPushButton("❌ Lỗi (0)")
        self.btn_filter_error.setCheckable(True)
        self.btn_filter_error.clicked.connect(lambda: self._apply_filter(CATEGORY_ERROR))
        filter_row.addWidget(self.btn_filter_error)

        filter_row.addStretch()

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔍 Tìm theo tên video hoặc lời thoại...")
        self.edit_search.setFixedWidth(280)
        self.edit_search.textChanged.connect(self._filter_table_rows)
        filter_row.addWidget(self.edit_search)

        tab_table_layout.addLayout(filter_row)

        # Table Widget: 11 Cột
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "#", "Tên Video", "Kích Thước", "Độ Phân Giải / Tỉ Lệ", "Phân Loại",
            "Số Từ", "Thời Lượng Voice", "Tốc Độ (t/s)", "Ngôn Ngữ", "Nội Dung Thoại / Chi Tiết", "Thao Tác"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)

        tab_table_layout.addWidget(self.table)
        self.tabs.addTab(tab_table, "📋 Danh Sách & Kết Quả Phân Loại")

        # TAB 2: Realtime Terminal Logs
        tab_log = QWidget()
        tab_log_layout = QVBoxLayout(tab_log)
        tab_log_layout.setContentsMargins(8, 8, 8, 8)
        tab_log_layout.setSpacing(6)

        log_ctrl_row = QHBoxLayout()
        self.chk_autoscroll = QCheckBox("Tự động cuộn xuống dòng mới")
        self.chk_autoscroll.setChecked(True)
        log_ctrl_row.addWidget(self.chk_autoscroll)
        log_ctrl_row.addStretch()

        btn_clear_log = QPushButton("Xóa nhật ký")
        btn_clear_log.setStyleSheet("background-color: #1E293B; border: 1px solid #334155;")
        btn_clear_log.clicked.connect(self._clear_logs)
        log_ctrl_row.addWidget(btn_clear_log)

        tab_log_layout.addLayout(log_ctrl_row)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setObjectName("logView")
        self.txt_log.setReadOnly(True)
        tab_log_layout.addWidget(self.txt_log)

        self.tabs.addTab(tab_log, "📝 Nhật Ký Hoạt Động (Logs)")

        root_layout.addWidget(self.tabs, stretch=1)

    def _create_metric_card(self, title: str, init_val: str, color_hex: str) -> QFrame:
        card = QFrame()
        card.setProperty("class", "metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 10px; font-weight: 700; color: #94A3B8; text-transform: uppercase;")
        lbl_val = QLabel(init_val)
        lbl_val.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {color_hex};")
        lbl_val.setObjectName("valLabel")

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        return card

    def _update_card_val(self, card: QFrame, val: str):
        lbl = card.findChild(QLabel, "valLabel")
        if lbl:
            lbl.setText(val)

    def _append_log(self, message: str, level: str = "info"):
        color_map = {
            "info": "#38BDF8",
            "success": "#34D399",
            "warn": "#FBBF24",
            "error": "#F87171"
        }
        color = color_map.get(level, "#E2E8F0")
        import datetime
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        html_msg = f'<span style="color: #64748B;">[{now_str}]</span> <span style="color: {color};">{message}</span>'
        self.txt_log.appendHtml(html_msg)

        if self.chk_autoscroll.isChecked():
            sb = self.txt_log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _clear_logs(self):
        self.txt_log.clear()

    # ================= FOLDER SCANNING =================
    def _browse_source_folder(self):
        cur = self.edit_source.text().strip() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa video", cur)
        if folder:
            self.edit_source.setText(folder)
            self._scan_source_folder(silent=False)

    def _on_source_text_changed(self, text: str):
        path = text.strip()
        if os.path.isdir(path):
            self.config["source_dir"] = path
            save_config(self.config)

    def _scan_source_folder(self, silent: bool = False):
        folder = self.edit_source.text().strip()
        if not folder or not os.path.isdir(folder):
            if not silent:
                QMessageBox.warning(self, "Lỗi thư mục", "Vui lòng chọn thư mục video hợp lệ!")
            return

        self.config["source_dir"] = folder
        save_config(self.config)

        self._append_log(f"🔍 Đang quét video trong: {folder}...", "info")
        self.scanned_videos = scan_video_files(folder, recursive=self.chk_recursive.isChecked())
        self.count_total = len(self.scanned_videos)
        self._update_card_val(self.card_total, str(self.count_total))

        # Reset counters
        self.count_unbox = 0
        self.count_review = 0
        self.count_deleted = 0
        self.count_doubt = 0
        self.count_error = 0
        self.results_map.clear()
        self._update_counts_ui()

        # Hiển thị lên bảng
        self._populate_initial_table()

        msg = f"Đã tìm thấy {self.count_total} video hợp lệ."
        self._append_log(f"✅ {msg}", "success")
        self.lbl_progress_info.setText(msg)
        self.prog_bar.setMaximum(max(1, self.count_total))
        self.prog_bar.setValue(0)
        self.prog_bar.setFormat(f"0 / {self.count_total} video (0%)")

    def _populate_initial_table(self):
        self.table.setRowCount(len(self.scanned_videos))
        for i, v in enumerate(self.scanned_videos):
            # 0: STT
            item_stt = QTableWidgetItem(str(i + 1))
            item_stt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, item_stt)

            # 1: Tên file
            item_name = QTableWidgetItem(v.file_name)
            item_name.setToolTip(v.file_path)
            self.table.setItem(i, 1, item_name)

            # 2: Kích thước
            item_size = QTableWidgetItem(f"{v.file_size_mb} MB")
            item_size.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(i, 2, item_size)

            # 3: Độ phân giải & Tỉ lệ
            item_specs = QTableWidgetItem("⏳ Chờ quét...")
            item_specs.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 3, item_specs)

            # 4: Phân loại
            item_cat = QTableWidgetItem("⏳ Đang chờ...")
            item_cat.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 4, item_cat)

            # 5: Số từ
            item_words = QTableWidgetItem("-")
            item_words.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 5, item_words)

            # 6: Thời lượng voice
            item_dur = QTableWidgetItem("-")
            item_dur.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 6, item_dur)

            # 7: Tốc độ
            item_speed = QTableWidgetItem("-")
            item_speed.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 7, item_speed)

            # 8: Ngôn ngữ
            item_lang = QTableWidgetItem("-")
            item_lang.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 8, item_lang)

            # 9: Lời thoại / Chi tiết
            item_text = QTableWidgetItem("-")
            self.table.setItem(i, 9, item_text)

            # 10: Thao tác
            btn_play = QPushButton("▶ Xem")
            btn_play.setStyleSheet("padding: 2px 6px; font-size: 11px;")
            path_copy = v.file_path
            btn_play.clicked.connect(lambda _, p=path_copy: self._open_file_in_player(p))
            self.table.setCellWidget(i, 10, btn_play)

    # ================= CLASSIFICATION EXECUTION =================
    def _start_classification(self):
        if not self.scanned_videos:
            self._scan_source_folder(silent=True)
            if not self.scanned_videos:
                QMessageBox.warning(self, "Chưa có video", "Vui lòng chọn thư mục chứa video trước khi bắt đầu!")
                return

        source_dir = self.edit_source.text().strip()
        dest_folders = {
            CATEGORY_UNBOX: os.path.join(source_dir, self.config.get("output_unbox", "1_Unbox_Khong_Voice")),
            CATEGORY_REVIEW: os.path.join(source_dir, self.config.get("output_review", "2_Review_Co_Voice")),
            CATEGORY_DOUBT: os.path.join(source_dir, self.config.get("output_doubt", "3_Can_Kiem_Tra_Lai"))
        }

        # Lưu cài đặt hiện tại
        self.config["workers"] = self.spin_workers.value()
        self.config["model_name"] = self.combo_quick_model.currentText()
        self.config["recursive"] = self.chk_recursive.isChecked()
        self.config["split_review_subtitles"] = False
        self.config["action_mode"] = "copy" if self.radio_copy.isChecked() else "move"
        self.config["delete_low_res"] = self.chk_del_low_res.isChecked()
        self.config["delete_non_9_16"] = self.chk_del_non_9_16.isChecked()
        self.config["delete_short_duration"] = self.chk_del_short.isChecked()
        save_config(self.config)

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸️ Tạm dừng")
        self.btn_stop.setEnabled(True)
        self.prog_bar.setValue(0)
        self.prog_bar.setMaximum(len(self.scanned_videos))

        self.worker = ClassifierWorker(self.scanned_videos, self.config, dest_folders)
        self.worker.sig_log.connect(self._append_log)
        self.worker.sig_progress.connect(self._on_worker_progress)
        self.worker.sig_result.connect(self._on_worker_result)
        self.worker.sig_finished.connect(self._on_worker_finished)
        self.worker.start()

    def _toggle_pause(self):
        if not self.worker:
            return
        if self.worker.is_paused():
            self.worker.resume()
            self.btn_pause.setText("⏸️ Tạm dừng")
        else:
            self.worker.pause()
            self.btn_pause.setText("▶️ Tiếp tục")

    def _stop_classification(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Xác nhận dừng",
                "Bạn có chắc chắn muốn dừng quét và phân loại không?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.btn_stop.setEnabled(False)

    def _on_worker_progress(self, current: int, total: int, current_file: str, elapsed_sec: float, eta_sec: float):
        self.prog_bar.setValue(current)
        pct = round((current / total) * 100) if total > 0 else 0
        self.prog_bar.setFormat(f"{current} / {total} video ({pct}%)")

        mins_el, secs_el = divmod(int(elapsed_sec), 60)
        mins_eta, secs_eta = divmod(int(eta_sec), 60)

        speed_min = round((current / elapsed_sec) * 60, 1) if elapsed_sec > 0 else 0
        self.lbl_progress_info.setText(
            f"Đang xử lý: {current_file} | Đã qua: {mins_el:02d}:{secs_el:02d} | Còn lại: ~{mins_eta:02d}:{secs_eta:02d} | Tốc độ: {speed_min} video/phút"
        )

    def _on_worker_result(self, res: ClassificationResult):
        self.results_map[res.file_name] = res

        # Cập nhật số liệu
        if res.category == CATEGORY_UNBOX:
            self.count_unbox += 1
        elif res.category == CATEGORY_UNBOX_MUSIC:
            self.count_unbox += 1
            self.count_unbox_music += 1
        elif res.category == CATEGORY_UNBOX_NOMUSIC:
            self.count_unbox += 1
            self.count_unbox_nomusic += 1
        elif res.category == CATEGORY_DELETED:
            self.count_deleted += 1
        elif res.category in (CATEGORY_REVIEW, CATEGORY_REVIEW_SUB, CATEGORY_REVIEW_NOSUB):
            self.count_review += 1
        elif res.category == CATEGORY_DOUBT:
            self.count_doubt += 1
        else:
            self.count_error += 1

        self._update_counts_ui()

        # Tìm dòng trong bảng để cập nhật
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            if name_item and name_item.text() == res.file_name:
                self._update_table_row(row, res)
                break

    def _update_table_row(self, row: int, res: ClassificationResult):
        # Cột 3: Độ Phân Giải / Tỉ Lệ
        specs_str = f"{res.resolution} ({res.aspect_ratio})" if res.resolution else "-"
        specs_item = QTableWidgetItem(specs_str)
        specs_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if res.category == CATEGORY_DELETED and ("< 720p" in res.category_name or "9:16" in res.category_name):
            specs_item.setForeground(QColor("#FCA5A5"))
        self.table.setItem(row, 3, specs_item)

        # Cột 4: Phân loại với badge màu
        cat_item = QTableWidgetItem(res.category_name)
        cat_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if res.category in (CATEGORY_REVIEW, CATEGORY_REVIEW_SUB, CATEGORY_REVIEW_NOSUB):
            cat_item.setForeground(QColor("#34D399"))
            cat_item.setBackground(QColor("#064E3B"))
        elif res.category in (CATEGORY_UNBOX, CATEGORY_UNBOX_MUSIC, CATEGORY_UNBOX_NOMUSIC):
            cat_item.setForeground(QColor("#38BDF8"))
            cat_item.setBackground(QColor("#0C4A6E"))
        elif res.category == CATEGORY_DELETED:
            cat_item.setForeground(QColor("#FCA5A5"))
            cat_item.setBackground(QColor("#450A0A"))
        elif res.category == CATEGORY_DOUBT:
            cat_item.setForeground(QColor("#FBBF24"))
            cat_item.setBackground(QColor("#78350F"))
        else:
            cat_item.setForeground(QColor("#F87171"))
            cat_item.setBackground(QColor("#7F1D1D"))
        self.table.setItem(row, 4, cat_item)

        # Cột 5: Số từ
        w_val = str(res.total_words) if res.category != CATEGORY_DELETED else "-"
        w_item = QTableWidgetItem(w_val)
        w_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 5, w_item)

        # Cột 6: Thời lượng voice / Thời lượng video
        dur_display = f"{res.total_speech_time}s" if res.category != CATEGORY_DELETED else f"{res.video_duration:.1f}s"
        d_item = QTableWidgetItem(dur_display)
        d_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if res.category == CATEGORY_DELETED and "10s" in res.category_name:
            d_item.setForeground(QColor("#FCA5A5"))
        self.table.setItem(row, 6, d_item)

        # Cột 7: Tốc độ
        s_val = f"{res.speech_rate} t/s" if res.category != CATEGORY_DELETED else "-"
        s_item = QTableWidgetItem(s_val)
        s_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 7, s_item)

        # Cột 8: Ngôn ngữ
        l_item = QTableWidgetItem(res.detected_language.upper())
        l_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 8, l_item)

        # Cột 9: Lời thoại preview / Lý do xóa
        t_item = QTableWidgetItem(res.sample_text)
        t_item.setToolTip(res.sample_text)
        if res.category == CATEGORY_DELETED:
            t_item.setForeground(QColor("#FCA5A5"))
        self.table.setItem(row, 9, t_item)

        # Cột 10: Thao tác
        if res.category == CATEGORY_DELETED:
            btn_play = QPushButton("🗑️ Đã xóa")
            btn_play.setStyleSheet("padding: 2px 6px; font-size: 11px; color: #94A3B8; background-color: #1E293B;")
            btn_play.setEnabled(False)
        else:
            target_path = res.destination_path or res.file_path
            btn_play = QPushButton("▶ Xem file")
            btn_play.setStyleSheet("padding: 2px 6px; font-size: 11px;")
            btn_play.clicked.connect(lambda _, p=target_path: self._open_file_in_player(p))
        self.table.setCellWidget(row, 10, btn_play)

    def _on_worker_finished(self, total_done: int, summary: dict):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        if "error" in summary:
            QMessageBox.critical(self, "Lỗi phân loại", f"Quá trình bị lỗi: {summary['error']}")
            return

        total_time = summary.get("total_time", 0)
        self.lbl_progress_info.setText(f"🎉 Hoàn thành {total_done} video trong {total_time}s!")
        QMessageBox.information(
            self, "Hoàn tất phân loại",
            f"🎉 Đã xử lý xong {total_done} video!\n\n"
            f"• 🎙️ Review (Có Voice): {self.count_review}\n"
            f"• 📦 Unbox (Không voice): {self.count_unbox} (🎵 Có nhạc: {self.count_unbox_music} | 🔇 Không nhạc: {self.count_unbox_nomusic})\n"
            f"• 🗑️ Đã xóa (Không đạt chuẩn Shopee): {self.count_deleted}\n"
            f"• ⚠️ Cần kiểm tra lại: {self.count_doubt}\n"
            f"• ❌ Lỗi: {self.count_error}\n\n"
            f"Thời gian xử lý: {total_time} giây."
        )

    def _update_counts_ui(self):
        self._update_card_val(self.card_review, str(self.count_review))
        self._update_card_val(self.card_unbox, f"{self.count_unbox}")
        self._update_card_val(self.card_deleted, str(self.count_deleted))
        self._update_card_val(self.card_doubt, str(self.count_doubt))
        self._update_card_val(self.card_error, str(self.count_error))

        # Cập nhật số trên các nút lọc
        self.btn_filter_all.setText(f"Tất cả ({self.count_total})")
        self.btn_filter_review.setText(f"🎙️ Review ({self.count_review})")
        self.btn_filter_unbox.setText(f"📦 Unbox ({self.count_unbox})")
        self.btn_filter_unbox_music.setText(f"🎵 Có Nhạc ({self.count_unbox_music})")
        self.btn_filter_unbox_nomusic.setText(f"🔇 Không Nhạc ({self.count_unbox_nomusic})")
        self.btn_filter_deleted.setText(f"🗑️ Đã xóa ({self.count_deleted})")
        self.btn_filter_doubt.setText(f"⚠️ Cần KT ({self.count_doubt})")
        self.btn_filter_error.setText(f"❌ Lỗi ({self.count_error})")

    # ================= FILTERS & SEARCH =================
    def _apply_filter(self, category: str):
        buttons = [
            ("ALL", self.btn_filter_all),
            (CATEGORY_REVIEW, self.btn_filter_review),
            (CATEGORY_UNBOX, self.btn_filter_unbox),
            (CATEGORY_UNBOX_MUSIC, self.btn_filter_unbox_music),
            (CATEGORY_UNBOX_NOMUSIC, self.btn_filter_unbox_nomusic),
            (CATEGORY_DELETED, self.btn_filter_deleted),
            (CATEGORY_DOUBT, self.btn_filter_doubt),
            (CATEGORY_ERROR, self.btn_filter_error)
        ]
        for cat, btn in buttons:
            btn.setChecked(cat == category)

        self._filter_table_rows()

    def _filter_table_rows(self):
        search_kw = self.edit_search.text().strip().lower()

        active_cat = "ALL"
        if self.btn_filter_review.isChecked():
            active_cat = CATEGORY_REVIEW
        elif self.btn_filter_unbox.isChecked():
            active_cat = CATEGORY_UNBOX
        elif self.btn_filter_unbox_music.isChecked():
            active_cat = CATEGORY_UNBOX_MUSIC
        elif self.btn_filter_unbox_nomusic.isChecked():
            active_cat = CATEGORY_UNBOX_NOMUSIC
        elif self.btn_filter_deleted.isChecked():
            active_cat = CATEGORY_DELETED
        elif self.btn_filter_doubt.isChecked():
            active_cat = CATEGORY_DOUBT
        elif self.btn_filter_error.isChecked():
            active_cat = CATEGORY_ERROR

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            file_name = name_item.text().lower() if name_item else ""

            text_item = self.table.item(row, 9)
            preview_text = text_item.text().lower() if text_item else ""

            res = self.results_map.get(name_item.text()) if name_item else None
            res_cat = res.category if res else None

            if active_cat == "ALL":
                cat_match = True
            elif active_cat == CATEGORY_REVIEW:
                cat_match = (res_cat in (CATEGORY_REVIEW, CATEGORY_REVIEW_SUB, CATEGORY_REVIEW_NOSUB))
            elif active_cat == CATEGORY_UNBOX:
                cat_match = (res_cat in (CATEGORY_UNBOX, CATEGORY_UNBOX_MUSIC, CATEGORY_UNBOX_NOMUSIC))
            else:
                cat_match = (res_cat == active_cat)

            kw_match = (search_kw == "") or (search_kw in file_name) or (search_kw in preview_text)

            self.table.setRowHidden(row, not (cat_match and kw_match))

    # ================= SETTINGS & OPTIONS =================
    def _open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.config = dialog.get_config()
            save_config(self.config)
            self.combo_quick_model.setCurrentText(self.config.get("model_name", "tiny"))
            if self.config.get("action_mode", "move") == "copy":
                self.radio_copy.setChecked(True)
            else:
                self.radio_move.setChecked(True)

            # Đồng bộ lại các checkbox nhanh
            self.chk_del_low_res.setChecked(self.config.get("delete_low_res", True))
            self.chk_del_non_9_16.setChecked(self.config.get("delete_non_9_16", True))
            self.chk_del_short.setChecked(self.config.get("delete_short_duration", True))

            self._append_log("⚙️ Đã lưu cấu hình tham số mới.", "info")

    def _on_model_changed(self, text: str):
        self.config["model_name"] = text
        save_config(self.config)

    def _on_option_changed(self):
        self.config["recursive"] = self.chk_recursive.isChecked()
        self.config["split_review_subtitles"] = False
        self.config["action_mode"] = "copy" if self.radio_copy.isChecked() else "move"
        self.config["workers"] = self.spin_workers.value()
        self.config["delete_low_res"] = self.chk_del_low_res.isChecked()
        self.config["delete_non_9_16"] = self.chk_del_non_9_16.isChecked()
        self.config["delete_short_duration"] = self.chk_del_short.isChecked()
        save_config(self.config)

    # ================= SYSTEM ACTIONS =================
    def _open_file_in_player(self, file_path: str):
        if os.path.exists(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
        else:
            QMessageBox.warning(self, "Không tìm thấy file", f"File không tồn tại: {file_path}")

    def _open_source_in_explorer(self):
        folder = self.edit_source.text().strip()
        if os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        else:
            QMessageBox.warning(self, "Lỗi thư mục", "Thư mục không tồn tại!")

    def _open_dest_folder(self, category: str):
        source = self.edit_source.text().strip()
        if not source or not os.path.isdir(source):
            QMessageBox.warning(self, "Chưa chọn thư mục gốc", "Vui lòng chọn thư mục gốc trước!")
            return

        review_base = os.path.join(source, self.config.get("output_review", "2_Review_Co_Voice"))
        unbox_base = os.path.join(source, self.config.get("output_unbox", "1_Unbox_Khong_Voice"))
        if category in (CATEGORY_REVIEW, CATEGORY_REVIEW_SUB, CATEGORY_REVIEW_NOSUB):
            target = review_base
        elif category == CATEGORY_UNBOX_MUSIC:
            target = os.path.join(unbox_base, self.config.get("output_unbox_music", "1_Co_Nhac"))
        elif category == CATEGORY_UNBOX_NOMUSIC:
            target = os.path.join(unbox_base, self.config.get("output_unbox_nomusic", "2_Khong_Nhac"))
        elif category == CATEGORY_UNBOX:
            target = unbox_base
        else:
            target = os.path.join(source, self.config.get("output_doubt", "3_Can_Kiem_Tra_Lai"))

        os.makedirs(target, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def _export_to_csv(self):
        if not self.results_map:
            QMessageBox.information(self, "Chưa có dữ liệu", "Chưa có video nào được phân loại để xuất báo cáo!")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Lưu báo cáo CSV", "Bao_Cao_Phan_Loai_Video.csv", "CSV Files (*.csv)")
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Tên File", "Phân Loại", "Mã Loại", "Độ Phân Giải", "Tỉ Lệ Khung Hình",
                    "Thời Lượng Video (s)", "Số Từ", "Thời Lượng Voice (s)",
                    "Tốc Độ (từ/s)", "Ngôn Ngữ", "Kích Thước (MB)", "Thời Gian Xử Lý (s)",
                    "Đường Dẫn Đích", "Nội Dung Thoại Whisper / Lý Do Xóa", "Lỗi Nếu Có"
                ])
                for res in self.results_map.values():
                    writer.writerow([
                        res.file_name,
                        res.category_name,
                        res.category,
                        res.resolution,
                        res.aspect_ratio,
                        res.video_duration,
                        res.total_words,
                        res.total_speech_time,
                        res.speech_rate,
                        res.detected_language,
                        res.file_size_mb,
                        res.processing_time,
                        res.destination_path,
                        res.sample_text,
                        res.error_message
                    ])
            self._append_log(f"📥 Đã xuất báo cáo CSV: {save_path}", "success")
            QMessageBox.information(self, "Xuất thành công", f"Đã lưu báo cáo thành công tại:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi xuất file", f"Không thể lưu file CSV: {e}")
