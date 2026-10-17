"""
Modern High-End Dark Theme Stylesheet for VoxSort AI (PyQt6)
"""

DARK_THEME_QSS = """
/* Global Window & Fonts */
QWidget {
    background-color: #0F172A;
    color: #F1F5F9;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    font-size: 13px;
    selection-background-color: #6366F1;
    selection-color: #FFFFFF;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #1E293B;
    width: 8px;
    margin: 0px 0px 0px 0px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #475569;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #64748B;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #1E293B;
    height: 8px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #475569;
    min-width: 20px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #64748B;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Card Containers & GroupBoxes */
QGroupBox {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    margin-top: 18px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    background-color: #312E81;
    color: #C7D2FE;
    border-radius: 6px;
    font-size: 12px;
}

/* Metric / Stats Cards */
QFrame.metricCard {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px;
}
QFrame.metricCard:hover {
    border: 1px solid #6366F1;
}

/* Input Fields */
QLineEdit {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 7px 10px;
    color: #F8FAFC;
    font-size: 13px;
}
QLineEdit:focus {
    border: 1px solid #6366F1;
    background-color: #131E33;
}
QLineEdit:read-only {
    background-color: #1E293B;
    color: #94A3B8;
}

/* ComboBox */
QComboBox {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    color: #F8FAFC;
    min-width: 90px;
}
QComboBox:focus, QComboBox:hover {
    border: 1px solid #6366F1;
}
QComboBox::drop-down {
    border: none;
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #1E293B;
    border: 1px solid #475569;
    border-radius: 6px;
    selection-background-color: #4F46E5;
    color: #F8FAFC;
    padding: 4px;
}

/* SpinBox */
QSpinBox, QDoubleSpinBox {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    color: #F8FAFC;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #6366F1;
}

/* Push Buttons */
QPushButton {
    background-color: #334155;
    color: #F8FAFC;
    border: 1px solid #475569;
    border-radius: 7px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #475569;
    border-color: #64748B;
}
QPushButton:pressed {
    background-color: #1E293B;
}
QPushButton:disabled {
    background-color: #1E293B;
    color: #64748B;
    border-color: #334155;
}

/* Primary Action Button (Start) */
QPushButton#btnStart {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366F1, stop:1 #8B5CF6);
    color: #FFFFFF;
    border: none;
    font-size: 14px;
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: 700;
}
QPushButton#btnStart:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4F46E5, stop:1 #7C3AED);
}
QPushButton#btnStart:disabled {
    background: #334155;
    color: #64748B;
}

/* Pause Button */
QPushButton#btnPause {
    background-color: #D97706;
    color: #FFFFFF;
    border: none;
    font-size: 13px;
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 600;
}
QPushButton#btnPause:hover {
    background-color: #B45309;
}

/* Stop Button */
QPushButton#btnStop {
    background-color: #DC2626;
    color: #FFFFFF;
    border: none;
    font-size: 13px;
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 600;
}
QPushButton#btnStop:hover {
    background-color: #B91C1C;
}

/* Secondary Action Button */
QPushButton.btnSecondary {
    background-color: #1E293B;
    border: 1px solid #334155;
    color: #CBD5E1;
}
QPushButton.btnSecondary:hover {
    background-color: #334155;
    color: #FFFFFF;
    border-color: #6366F1;
}

/* Radio Button & Checkbox */
QRadioButton, QCheckBox {
    color: #E2E8F0;
    spacing: 8px;
}
QRadioButton::indicator, QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #64748B;
    background-color: #0F172A;
    border-radius: 4px;
}
QRadioButton::indicator {
    border-radius: 8px;
}
QRadioButton::indicator:checked, QCheckBox::indicator:checked {
    background-color: #6366F1;
    border-color: #818CF8;
}

/* Progress Bar */
QProgressBar {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    text-align: center;
    color: #F8FAFC;
    font-weight: bold;
    height: 22px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:1 #06B6D4);
    border-radius: 7px;
}

/* Table Widget */
QTableWidget {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    gridline-color: #334155;
    color: #F8FAFC;
    selection-background-color: #312E81;
    selection-color: #FFFFFF;
}
QTableWidget::item {
    padding: 6px;
    border-bottom: 1px solid #1E293B;
}
QTableWidget::item:selected {
    background-color: #3730A3;
}
QHeaderView::section {
    background-color: #0F172A;
    color: #94A3B8;
    font-weight: 700;
    font-size: 12px;
    padding: 8px 6px;
    border: none;
    border-bottom: 2px solid #334155;
    border-right: 1px solid #1E293B;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #334155;
    background-color: #1E293B;
    border-radius: 8px;
}
QTabBar::tab {
    background-color: #0F172A;
    color: #94A3B8;
    padding: 8px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #1E293B;
    color: #6366F1;
    border: 1px solid #334155;
    border-bottom: none;
}
QTabBar::tab:hover:!selected {
    color: #F8FAFC;
    background-color: #192339;
}

/* Log Text Edit */
QPlainTextEdit#logView {
    background-color: #0B0F19;
    color: #34D399;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px;
}

/* Status Badges Colors */
.badgeReview {
    background-color: #064E3B;
    color: #34D399;
    font-weight: bold;
    border-radius: 4px;
    padding: 3px 8px;
}
.badgeUnbox {
    background-color: #0C4A6E;
    color: #38BDF8;
    font-weight: bold;
    border-radius: 4px;
    padding: 3px 8px;
}
.badgeDoubt {
    background-color: #78350F;
    color: #FBBF24;
    font-weight: bold;
    border-radius: 4px;
    padding: 3px 8px;
}
.badgeError {
    background-color: #7F1D1D;
    color: #F87171;
    font-weight: bold;
    border-radius: 4px;
    padding: 3px 8px;
}
"""
