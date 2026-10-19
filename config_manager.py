import json
import os
from typing import Any, Dict

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    "source_dir": "",
    "output_unbox": "1_Unbox_Khong_Voice",
    "output_unbox_music": "1_Co_Nhac",
    "output_unbox_nomusic": "2_Khong_Nhac",
    "output_review": "2_Review_Co_Voice",
    "output_review_sub": "1_Co_Phu_De",
    "output_review_nosub": "2_Khong_Phu_De",
    "output_doubt": "3_Can_Kiem_Tra_Lai",
    "split_review_subtitles": False,
    "ocr_sample_frames": 6,
    "action_mode": "copy",  # "move" or "copy"
    "model_name": "tiny",   # tiny, base, small, medium
    "device": "cpu",
    "compute_type": "int8",
    "workers": 2,
    "unbox_max_words": 8,
    "unbox_max_duration": 2.0,
    "review_min_words": 15,
    "review_min_speed": 1.5,
    "no_speech_prob": 0.6,
    "beam_size": 1,
    "scan_duration_limit": 0,  # 0 = scan all, or seconds (e.g. 60)
    "recursive": False,
    # --- BỘ LỌC TỰ ĐỘNG XÓA MẶC ĐỊNH (CHUẨN SHOPEE) ---
    "delete_low_res": True,             # Mặc định XÓA video có độ phân giải dưới 720p
    "filter_min_resolution": 720,       # Ngưỡng phân giải tối thiểu (720p)
    "delete_non_9_16": True,            # Mặc định XÓA video không phải tỉ lệ 9:16
    "filter_aspect_ratio": "9:16",      # Chuẩn tỉ lệ 9:16
    "delete_short_duration": True,      # Mặc định XÓA video có thời lượng dưới 10s
    "filter_min_duration": 10.0         # Thời lượng tối thiểu (10 giây)
}

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json with fallback to default values."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                config = DEFAULT_CONFIG.copy()
                config.update(data)
                return config
        except Exception as e:
            print(f"Lỗi khi đọc config.json: {e}")
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to config.json."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Lỗi khi lưu config.json: {e}")
        return False
