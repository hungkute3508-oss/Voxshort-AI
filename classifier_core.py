import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import av
import cv2

# Suppress symlink warning on Windows for huggingface hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".wmv")

CATEGORY_UNBOX = "UNBOX"
CATEGORY_REVIEW = "REVIEW"
CATEGORY_REVIEW_SUB = "REVIEW_SUB"
CATEGORY_REVIEW_NOSUB = "REVIEW_NOSUB"
CATEGORY_DOUBT = "NGHI_NGO"
CATEGORY_ERROR = "ERROR"
CATEGORY_DELETED = "DELETED"

CATEGORY_NAMES = {
    CATEGORY_UNBOX: "📦 Unbox (Không Voice)",
    CATEGORY_REVIEW: "🎙️ Review (Có Voice)",
    CATEGORY_REVIEW_SUB: "🟢 Review (Có Phụ Đề)",
    CATEGORY_REVIEW_NOSUB: "🔵 Review (Không Phụ Đề)",
    CATEGORY_DOUBT: "⚠️ Cần kiểm tra lại",
    CATEGORY_ERROR: "❌ Lỗi xử lý",
    CATEGORY_DELETED: "🗑️ Đã xóa (Kém chuẩn)"
}

# Lazy global OCR instance
_GLOBAL_OCR = None

def get_ocr_instance():
    """Khởi tạo một instance RapidOCR duy nhất trong RAM để tái sử dụng."""
    global _GLOBAL_OCR
    if _GLOBAL_OCR is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _GLOBAL_OCR = RapidOCR()
        except Exception as e:
            print(f"Lỗi khởi tạo RapidOCR: {e}")
            _GLOBAL_OCR = None
    return _GLOBAL_OCR

@dataclass
class VideoInfo:
    file_name: str
    file_path: str
    file_size_mb: float

@dataclass
class ClassificationResult:
    file_name: str
    file_path: str
    category: str
    category_name: str
    total_words: int
    total_speech_time: float
    speech_rate: float
    detected_language: str
    sample_text: str
    destination_path: str
    file_size_mb: float
    processing_time: float
    has_subtitles: bool = False
    subtitle_sample: str = ""
    error_message: str = ""
    resolution: str = ""
    aspect_ratio: str = ""
    video_duration: float = 0.0

def get_unique_filename(dest_folder: str, file_name: str) -> str:
    """Đảm bảo không bị trùng tên file khi copy hoặc move vào thư mục đích."""
    target_path = os.path.join(dest_folder, file_name)
    if not os.path.exists(target_path):
        return target_path

    base_name, ext = os.path.splitext(file_name)
    counter = 1
    while True:
        new_name = f"{base_name} ({counter}){ext}"
        new_path = os.path.join(dest_folder, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def get_video_specs(video_path: str) -> Tuple[int, int, float, str]:
    """
    Trích xuất kích thước (width, height), thời lượng (duration) và tỉ lệ khung hình của video.
    Hỗ trợ xử lý xoay màn hình (rotation metadata) và fallback PyAV + OpenCV.
    """
    w, h, duration = 0, 0, 0.0
    # 1. Thử đọc nhanh bằng PyAV (nhanh gấp 3 lần, không cần decode frame)
    try:
        with av.open(video_path) as container:
            if container.duration:
                duration = float(container.duration / av.time_base)
            v_stream = next((s for s in container.streams.video), None)
            if v_stream:
                w = v_stream.codec_context.width or v_stream.width
                h = v_stream.codec_context.height or v_stream.height
                if duration <= 0 and v_stream.duration and v_stream.time_base:
                    duration = float(v_stream.duration * v_stream.time_base)
                # Kiểm tra xoay màn hình (rotate tag)
                rot = v_stream.metadata.get("rotate", 0)
                try:
                    rot = int(rot)
                except Exception:
                    rot = 0
                if rot in (90, 270):
                    w, h = h, w
    except Exception:
        pass

    # 2. Fallback sang OpenCV nếu PyAV thiếu thông tin
    if w <= 0 or h <= 0 or duration <= 0:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                if w <= 0:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if h <= 0:
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if duration <= 0:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    if fps > 0 and fc > 0:
                        duration = fc / fps
            cap.release()
        except Exception:
            pass

    # Xác định tỉ lệ hiển thị
    if h > 0 and w > 0:
        r = w / h
        if abs(r - (9.0 / 16.0)) <= 0.03 and h > w:
            ratio_str = "9:16"
        elif abs(r - (16.0 / 9.0)) <= 0.03 and w > h:
            ratio_str = "16:9"
        elif abs(r - 1.0) <= 0.03:
            ratio_str = "1:1"
        elif abs(r - 0.75) <= 0.03:
            ratio_str = "3:4"
        else:
            ratio_str = f"{r:.2f}"
    else:
        ratio_str = "unknown"

    return w, h, duration, ratio_str

def check_video_shopee_filters(
    video_path: str,
    config: Dict[str, Any]
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Kiểm tra 3 tiêu chuẩn lọc video mặc định:
    1. Xóa video có độ phân giải dưới 720p (ngưỡng min_res = 720)
    2. Xóa video có định dạng không phải 9:16 (video dọc chuẩn Shopee/TikTok)
    3. Xóa video có thời lượng dưới 10s (ngưỡng min_dur = 10.0s)

    Trả về: (should_delete, reasons_short, reasons_detail, specs)
    """
    t0 = time.time()
    w, h, duration, ratio_str = get_video_specs(video_path)
    elapsed = round(time.time() - t0, 3)

    specs = {
        "width": w,
        "height": h,
        "duration": round(duration, 2),
        "ratio_str": ratio_str,
        "check_time": elapsed
    }

    reasons_short: List[str] = []
    reasons_detail: List[str] = []

    delete_low_res = config.get("delete_low_res", True)
    min_res = config.get("filter_min_resolution", 720)

    delete_non_9_16 = config.get("delete_non_9_16", True)

    delete_short_duration = config.get("delete_short_duration", True)
    min_dur = config.get("filter_min_duration", 10.0)

    # 1. Kiểm tra độ phân giải < 720p
    # Chiều ngắn (short edge) của video dưới 720p
    if delete_low_res:
        short_edge = min(w, h) if (w > 0 and h > 0) else 0
        if short_edge < min_res:
            reasons_short.append(f"< {min_res}p")
            reasons_detail.append(f"Độ phân giải {w}x{h} (< {min_res}p)")

    # 2. Kiểm tra định dạng 9:16
    if delete_non_9_16:
        is_9_16 = False
        if w > 0 and h > 0 and h > w:
            ratio = w / h
            if abs(ratio - (9.0 / 16.0)) <= 0.03:
                is_9_16 = True

        if not is_9_16:
            reasons_short.append("Không phải 9:16")
            reasons_detail.append(f"Tỉ lệ khung hình {w}:{h} ({ratio_str}) không phải 9:16")

    # 3. Kiểm tra thời lượng < 10s
    if delete_short_duration:
        if duration < min_dur:
            reasons_short.append(f"< {int(min_dur)}s")
            reasons_detail.append(f"Thời lượng {duration:.1f}s (< {min_dur:.0f}s)")

    should_delete = len(reasons_short) > 0
    return should_delete, reasons_short, reasons_detail, specs

def delete_video_file(file_path: str) -> Tuple[bool, str]:
    """
    Xóa file video khỏi ổ đĩa một cách an toàn và có cơ chế thử lại nếu bị khóa trên Windows.
    """
    if not os.path.exists(file_path):
        return False, "File không tồn tại"

    for attempt in range(3):
        try:
            os.remove(file_path)
            return True, ""
        except PermissionError:
            time.sleep(0.15)
        except Exception as e:
            return False, str(e)

    try:
        os.remove(file_path)
        return True, ""
    except Exception as e:
        return False, f"Không thể xóa file (file đang bị khóa): {e}"

def check_has_audio(video_path: str) -> bool:
    """Kiểm tra video có chứa luồng âm thanh hay không."""
    try:
        with av.open(video_path) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return True

def detect_subtitles_ocr(video_path: str, num_samples: int = 6) -> Tuple[bool, str]:
    """
    Quét nhận diện phụ đề (hardcoded/burned-in subtitles) trên khung hình video.
    Lấy mẫu num_samples mốc thời gian trải đều từ 15% đến 85% thời lượng.
    Tối ưu siêu nhanh (~0.15s) bằng cách crop vùng phụ đề (35% - 92% chiều cao) và resize.
    """
    ocr = get_ocr_instance()
    if ocr is None:
        return False, ""

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, ""

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps
        if duration < 1.0 or total_frames < 10:
            cap.release()
            return False, ""

        # Lấy mẫu các mốc thời gian trải đều
        sample_ratios = [0.15 + i * (0.7 / max(1, num_samples - 1)) for i in range(num_samples)]
        frames_with_text = 0
        unique_texts = set()
        snippets = []

        for ratio in sample_ratios:
            frame_idx = int(total_frames * ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            # Vùng phụ đề: từ 35% đến 92% chiều cao màn hình (bỏ qua header trên cùng và thanh điều hướng đáy)
            sub_roi = frame[int(h * 0.35):int(h * 0.92), :]

            # Giảm độ phân giải để OCR chạy siêu tốc (dưới 25ms/frame)
            if w > 720:
                scale = 720.0 / w
                sub_roi = cv2.resize(sub_roi, (720, int(sub_roi.shape[0] * scale)))

            result, _ = ocr(sub_roi)
            if result:
                valid_texts = [r[1].strip() for r in result if len(r[1].strip()) >= 3]
                if valid_texts:
                    frames_with_text += 1
                    combined = " ".join(valid_texts)
                    unique_texts.add(combined.lower())
                    if len(snippets) < 2:
                        snippets.append(f"{int(ratio * duration)}s: {combined[:35]}")

        cap.release()

        # Tiêu chuẩn có phụ đề:
        # 1. Chữ xuất hiện ở ít nhất 35% số frame mẫu (ví dụ >= 2 trong 6 frame).
        # 2. Text phải thay đổi giữa các frame (len(unique_texts) >= 2) để loại trừ watermark/logo cố định.
        min_required_frames = max(2, int(num_samples * 0.35))
        has_sub = (frames_with_text >= min_required_frames) and (len(unique_texts) >= 2)
        sample_str = " | ".join(snippets) if snippets else ""
        return has_sub, sample_str

    except Exception as e:
        print(f"Lỗi khi OCR phụ đề file {video_path}: {e}")
        return False, ""

def scan_video_files(source_dir: str, recursive: bool = False) -> List[VideoInfo]:
    """Quét toàn bộ danh sách file video hợp lệ trong thư mục."""
    video_list: List[VideoInfo] = []
    if not source_dir or not os.path.exists(source_dir):
        return video_list

    if recursive:
        for root, _, files in os.walk(source_dir):
            for f in files:
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    full_path = os.path.join(root, f)
                    try:
                        size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)
                    except Exception:
                        size_mb = 0.0
                    video_list.append(VideoInfo(file_name=f, file_path=full_path, file_size_mb=size_mb))
    else:
        try:
            for f in os.listdir(source_dir):
                full_path = os.path.join(source_dir, f)
                if os.path.isfile(full_path) and f.lower().endswith(SUPPORTED_EXTENSIONS):
                    try:
                        size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)
                    except Exception:
                        size_mb = 0.0
                    video_list.append(VideoInfo(file_name=f, file_path=full_path, file_size_mb=size_mb))
        except Exception as e:
            print(f"Lỗi khi quét thư mục {source_dir}: {e}")

    return video_list

def transcribe_and_classify(
    model,
    video_path: str,
    config: Dict[str, Any]
) -> ClassificationResult:
    """Phân tích âm thanh trong video bằng Whisper và phân loại theo tiêu chí."""
    start_time = time.time()
    file_name = os.path.basename(video_path)
    try:
        file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2)
    except Exception:
        file_size_mb = 0.0

    unbox_max_words = config.get("unbox_max_words", 8)
    unbox_max_dur = config.get("unbox_max_duration", 2.0)
    review_min_words = config.get("review_min_words", 15)
    review_min_speed = config.get("review_min_speed", 1.5)
    no_speech_threshold = config.get("no_speech_prob", 0.6)
    beam_size = config.get("beam_size", 1)
    scan_limit = config.get("scan_duration_limit", 0)
    split_subtitles = config.get("split_review_subtitles", False)
    ocr_samples = config.get("ocr_sample_frames", 6)

    # 1. Kiểm tra luồng âm thanh trước để tối ưu tốc độ
    if not check_has_audio(video_path):
        elapsed = round(time.time() - start_time, 2)
        return ClassificationResult(
            file_name=file_name,
            file_path=video_path,
            category=CATEGORY_UNBOX,
            category_name=CATEGORY_NAMES[CATEGORY_UNBOX],
            total_words=0,
            total_speech_time=0.0,
            speech_rate=0.0,
            detected_language="none",
            sample_text="(Video không có luồng âm thanh)",
            destination_path="",
            file_size_mb=file_size_mb,
            processing_time=elapsed,
            error_message=""
        )

    try:
        # Transcribe bằng Faster-Whisper
        segments, info = model.transcribe(
            video_path,
            vad_filter=True,
            beam_size=beam_size
        )

        tong_so_tu = 0
        tong_thoi_gian = 0.0
        sample_texts: List[str] = []
        is_asian_lang = info.language in ['zh', 'ja', 'ko'] if info and info.language else False

        for segment in segments:
            if scan_limit > 0 and segment.start >= scan_limit:
                break

            text = segment.text.strip()
            if not text:
                continue

            if segment.no_speech_prob > no_speech_threshold:
                continue

            text_lower = text.lower()
            so_tu = len(text_lower) if is_asian_lang else len(text_lower.split())
            tong_so_tu += so_tu
            tong_thoi_gian += (segment.end - segment.start)

            if len(sample_texts) < 3:
                sample_texts.append(text)

        toc_do = (tong_so_tu / tong_thoi_gian) if tong_thoi_gian > 0 else 0.0

        # Phân loại theo thuật toán ban đầu
        is_review = False
        if tong_so_tu < unbox_max_words or tong_thoi_gian < unbox_max_dur:
            category = CATEGORY_UNBOX
        elif tong_so_tu >= review_min_words and toc_do >= review_min_speed:
            is_review = True
            category = CATEGORY_REVIEW
        else:
            category = CATEGORY_DOUBT

        has_sub = False
        sub_sample = ""

        # Nếu là REVIEW và người dùng bật tách phụ đề
        if is_review and split_subtitles:
            has_sub, sub_sample = detect_subtitles_ocr(video_path, num_samples=ocr_samples)
            if has_sub:
                category = CATEGORY_REVIEW_SUB
            else:
                category = CATEGORY_REVIEW_NOSUB

        sample_preview = " | ".join(sample_texts)
        if len(sample_preview) > 100:
            sample_preview = sample_preview[:97] + "..."

        elapsed = round(time.time() - start_time, 2)
        return ClassificationResult(
            file_name=file_name,
            file_path=video_path,
            category=category,
            category_name=CATEGORY_NAMES.get(category, category),
            total_words=tong_so_tu,
            total_speech_time=round(tong_thoi_gian, 2),
            speech_rate=round(toc_do, 2),
            detected_language=info.language if info and info.language else "unknown",
            sample_text=sample_preview if sample_preview else "(Không có lời thoại)",
            destination_path="",
            file_size_mb=file_size_mb,
            processing_time=elapsed,
            has_subtitles=has_sub,
            subtitle_sample=sub_sample,
            error_message=""
        )

    except IndexError:
        elapsed = round(time.time() - start_time, 2)
        return ClassificationResult(
            file_name=file_name,
            file_path=video_path,
            category=CATEGORY_UNBOX,
            category_name=CATEGORY_NAMES[CATEGORY_UNBOX],
            total_words=0,
            total_speech_time=0.0,
            speech_rate=0.0,
            detected_language="none",
            sample_text="(Âm thanh rỗng / Không có tiếng)",
            destination_path="",
            file_size_mb=file_size_mb,
            processing_time=elapsed,
            error_message=""
        )

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        err_msg = str(e)
        return ClassificationResult(
            file_name=file_name,
            file_path=video_path,
            category=CATEGORY_DOUBT,
            category_name=CATEGORY_NAMES[CATEGORY_DOUBT],
            total_words=0,
            total_speech_time=0.0,
            speech_rate=0.0,
            detected_language="error",
            sample_text="",
            destination_path="",
            file_size_mb=file_size_mb,
            processing_time=elapsed,
            error_message=err_msg
        )

def dispatch_file_action(
    result: ClassificationResult,
    config: Dict[str, Any],
    dest_folders: Dict[str, str]
) -> str:
    """Di chuyển hoặc sao chép file tới thư mục phân loại tương ứng."""
    if result.category == CATEGORY_DELETED:
        return ""

    if result.category == CATEGORY_ERROR:
        target_dir = dest_folders.get(CATEGORY_DOUBT, "")
    elif result.category == CATEGORY_REVIEW_SUB:
        # Thư mục con Có Phụ Đề bên trong 2_Review_Co_Voice
        base_review_dir = dest_folders.get(CATEGORY_REVIEW, "")
        sub_folder = config.get("output_review_sub", "1_Co_Phu_De")
        target_dir = os.path.join(base_review_dir, sub_folder)
    elif result.category == CATEGORY_REVIEW_NOSUB:
        # Thư mục con Không Phụ Đề bên trong 2_Review_Co_Voice
        base_review_dir = dest_folders.get(CATEGORY_REVIEW, "")
        nosub_folder = config.get("output_review_nosub", "2_Khong_Phu_De")
        target_dir = os.path.join(base_review_dir, nosub_folder)
    else:
        target_dir = dest_folders.get(result.category, "")

    if not target_dir:
        return result.file_path

    os.makedirs(target_dir, exist_ok=True)
    target_path = get_unique_filename(target_dir, result.file_name)

    action = config.get("action_mode", "move")
    try:
        if action == "move":
            shutil.move(result.file_path, target_path)
        else:
            shutil.copy2(result.file_path, target_path)
        result.destination_path = target_path
        return target_path
    except Exception as e:
        result.error_message = f"Lỗi di chuyển/sao chép: {e}"
        return result.file_path
