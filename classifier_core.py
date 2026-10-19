import os
import shutil
import time
import threading
import urllib.request
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import av
import cv2
import numpy as np
import scipy.signal as signal

# Suppress symlink warning on Windows for huggingface hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".wmv")

CATEGORY_UNBOX = "UNBOX"
CATEGORY_UNBOX_MUSIC = "UNBOX_MUSIC"
CATEGORY_UNBOX_NOMUSIC = "UNBOX_NOMUSIC"
CATEGORY_REVIEW = "REVIEW"
CATEGORY_REVIEW_SUB = "REVIEW_SUB"
CATEGORY_REVIEW_NOSUB = "REVIEW_NOSUB"
CATEGORY_DOUBT = "NGHI_NGO"
CATEGORY_ERROR = "ERROR"
CATEGORY_DELETED = "DELETED"

CATEGORY_NAMES = {
    CATEGORY_UNBOX: "📦 Unbox (Không Voice)",
    CATEGORY_UNBOX_MUSIC: "🎵 Unbox (Có Nhạc)",
    CATEGORY_UNBOX_NOMUSIC: "🔇 Unbox (Không Nhạc)",
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
    has_music: bool = False
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

# ================= CÁC MÔ HÌNH NHẬN DIỆN CON NGƯỜI & ÂM THANH =================
_thread_local = threading.local()

def ensure_model_in_cache(model_filename: str) -> str:
    """
    Đảm bảo file model nằm ở đường dẫn an toàn (~/.cache/voxsort/models).
    Điều này giải quyết triệt để lỗi OpenCV C++ không đọc được file trên Windows
    khi đường dẫn thư mục có dấu tiếng Việt (như 'Tool làm video shopee').
    """
    cache_dir = os.path.expanduser("~/.cache/voxsort/models")
    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.join(cache_dir, model_filename)

    # 1. Thử copy từ thư mục models nội bộ của project nếu có
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_filename)
    if os.path.exists(local_path):
        try:
            if not os.path.exists(target_path) or os.path.getsize(target_path) != os.path.getsize(local_path):
                shutil.copy2(local_path, target_path)
        except Exception:
            pass

    # 2. Nếu chưa có, tải tự động từ CDN uy tín
    if not os.path.exists(target_path) or os.path.getsize(target_path) < 10000:
        url_map = {
            "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "person_detection_mediapipe_2023mar_int8bq.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/person_detection_mediapipe/person_detection_mediapipe_2023mar_int8bq.onnx"
        }
        url = url_map.get(model_filename)
        if url:
            try:
                urllib.request.urlretrieve(url, target_path)
            except Exception as e:
                print(f"Không thể tải model {model_filename}: {e}")

    return target_path

def get_thread_human_detectors():
    """Lấy instances FaceDetectorYN và MPPersonDet riêng biệt cho từng luồng (Thread-safe)."""
    if not hasattr(_thread_local, "human_detectors"):
        yunet_path = ensure_model_in_cache("face_detection_yunet_2023mar.onnx")
        persondet_path = ensure_model_in_cache("person_detection_mediapipe_2023mar_int8bq.onnx")
        anchors_path = ensure_model_in_cache("mp_anchors.npy")

        yunet = None
        persondet = None
        anchors = None

        if os.path.exists(yunet_path) and os.path.getsize(yunet_path) > 100000:
            try:
                yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3)
            except Exception as e:
                print(f"Lỗi khởi tạo YuNet: {e}")

        if os.path.exists(persondet_path) and os.path.getsize(persondet_path) > 1000000:
            try:
                persondet = cv2.dnn.readNet(persondet_path)
            except Exception as e:
                print(f"Lỗi khởi tạo PersonDet: {e}")

        if os.path.exists(anchors_path):
            try:
                anchors = np.load(anchors_path)
            except Exception:
                pass

        _thread_local.human_detectors = (yunet, persondet, anchors)
    return _thread_local.human_detectors

def detect_human_in_video(video_path: str, num_samples: int = 6, config: Dict[str, Any] = None) -> Tuple[bool, str, float]:
    """
    Quét nhận diện sự xuất hiện của con người (khuôn mặt / dáng người) trong video.
    Lấy mẫu num_samples mốc thời gian trải đều.
    Trả về: (has_human, detail_reason, detected_time_sec)
    """
    yunet, persondet, anchors = get_thread_human_detectors()
    if yunet is None and persondet is None:
        return False, "", 0.0

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "", 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps
        if total_frames < 5:
            cap.release()
            return False, "", 0.0

        sample_ratios = [0.10 + i * (0.80 / max(1, num_samples - 1)) for i in range(num_samples)]

        for ratio in sample_ratios:
            frame_idx = int(total_frames * ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_time = round(ratio * duration, 1)
            h, w = frame.shape[:2]

            # 1. Kiểm tra khuôn mặt với YuNet (Chỉ nhận diện khuôn mặt người thật, bỏ qua bàn tay, móng tay, logo in nhỏ)
            if yunet is not None:
                in_w, in_h = w, h
                scale = 1.0
                if max(h, w) > 640:
                    scale = 640.0 / max(h, w)
                    in_w = int(w * scale)
                    in_h = int(h * scale)
                    frame_face = cv2.resize(frame, (in_w, in_h))
                else:
                    frame_face = frame

                yunet.setInputSize((in_w, in_h))
                _, faces = yunet.detect(frame_face)
                if faces is not None and len(faces) > 0:
                    for face in faces:
                        score = float(face[-1])
                        box_w = float(face[2])
                        box_h = float(face[3])

                        # Ngưỡng tin cậy >= 0.65 (mặt người thật rõ ràng)
                        if score < 0.65:
                            continue

                        # Kích thước mặt phải đủ rõ trên khung hình (ít nhất 5% chiều cao và 28px)
                        # Giúp bỏ qua các icon siêu nhỏ, bao bì in mờ, hoặc vân tay phản chiếu
                        if (box_h / in_h < 0.05) or (box_w / in_w < 0.05) or box_h < 28 or box_w < 28:
                            continue

                        # Tỉ lệ khung bao khuôn mặt người tự nhiên (rộng/cao từ 0.50 đến 1.45)
                        aspect = box_w / (box_h + 1e-6)
                        if aspect < 0.50 or aspect > 1.45:
                            continue

                        # Kiểm tra giải phẫu 5 điểm mốc (Landmarks: 2 mắt, mũi, 2 khóe miệng)
                        # Đảm bảo là khuôn mặt thật chứ không phải vân gỗ, góc nhăn vải hay nếp bàn tay
                        re_x, re_y = face[4], face[5]
                        le_x, le_y = face[6], face[7]
                        rc_x, rc_y = face[10], face[11]
                        lc_x, lc_y = face[12], face[13]

                        eye_y = (re_y + le_y) / 2.0
                        mouth_y = (rc_y + lc_y) / 2.0
                        eye_dist = np.sqrt((re_x - le_x) ** 2 + (re_y - le_y) ** 2)

                        # Miệng phải nằm bên dưới mắt và khoảng cách 2 mắt hợp lý
                        if mouth_y > eye_y + 3.0 and (box_w * 0.15 <= eye_dist <= box_w * 0.85):
                            cap.release()
                            return True, f"Phát hiện khuôn mặt người ở giây {frame_time}s (độ tin cậy: {int(score*100)}%)", frame_time

            # 2. Kiểm tra dáng người với MediaPipe PersonDet
            # CHỈ nhận diện khi xuất hiện cả người / thân người (torso/full body)
            # NẾU chỉ có cánh tay / bàn tay cầm nắm sản phẩm thì KHÔNG kích hoạt (để giữ lại video lọc bình thường)
            if persondet is not None and anchors is not None:
                scale_pd = max(h, w)
                pad_h = scale_pd - h
                pad_w = scale_pd - w
                padded = cv2.copyMakeBorder(frame, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                resized = cv2.resize(padded, (224, 224))
                blob = cv2.dnn.blobFromImage(resized, 2.0 / 255.0, (224, 224), (127.5, 127.5, 127.5), swapRB=True)
                persondet.setInput(blob)
                out = persondet.forward(persondet.getUnconnectedOutLayersNames())

                score_raw = out[0][0, :, 0].astype(np.float64)
                scores = 1.0 / (1.0 + np.exp(-np.clip(score_raw, -100, 100)))

                box_delta = out[1][0, :, 0:4]
                cxy = box_delta[:, :2] / 224.0
                wh = box_delta[:, 2:] / 224.0
                xy1 = (cxy - wh / 2.0 + anchors) * scale_pd
                xy2 = (cxy + wh / 2.0 + anchors) * scale_pd
                boxes = np.concatenate([xy1, xy2], axis=1)
                boxes -= [pad_w // 2, pad_h // 2, pad_w // 2, pad_h // 2]

                keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.50, 0.3)
                if len(keep) > 0:
                    for idx in keep:
                        sc = float(scores[idx])
                        if sc < 0.60:
                            continue

                        bx1, by1, bx2, by2 = boxes[idx]
                        bw = bx2 - bx1
                        bh = by2 - by1

                        # Tiêu chuẩn xác nhận là TOÀN THÂN hoặc THÂN NGƯỜI (không phải cánh tay/bàn tay đơn lẻ):
                        # 1. Chiều cao dáng người phải chiếm ít nhất 18% chiều cao khung hình
                        if bh / h < 0.18:
                            continue

                        # 2. Diện tích khung chiếm ít nhất 3.5% diện tích màn hình
                        if (bw * bh) / (w * h) < 0.035:
                            continue

                        # 3. Dáng người đứng/ngồi có tỉ lệ dọc hợp lý (không phải cánh tay thò ngang bàn bw/bh > 2.0)
                        if bw / (bh + 1e-6) > 2.0:
                            continue

                        cap.release()
                        return True, f"Phát hiện dáng người xuất hiện ở giây {frame_time}s (độ tin cậy: {int(sc*100)}%)", frame_time

        cap.release()
        return False, "", 0.0

    except Exception as e:
        print(f"Lỗi khi nhận diện con người file {video_path}: {e}")
        return False, "", 0.0

def extract_audio_samples(video_path: str, max_duration: float = 30.0, target_sr: int = 16000):
    """Trích xuất mảng audio 16kHz mono từ video bằng PyAV cực nhanh."""
    try:
        with av.open(video_path) as container:
            if len(container.streams.audio) == 0:
                return None, 0.0
            audio_stream = container.streams.audio[0]
            resampler = av.AudioResampler(format='flt', layout='mono', rate=target_sr)
            samples_list = []
            total_time = 0.0
            for frame in container.decode(audio_stream):
                resampled_frames = resampler.resample(frame)
                if resampled_frames:
                    for rf in resampled_frames:
                        arr = rf.to_ndarray()[0]
                        samples_list.append(arr)
                        total_time += len(arr) / target_sr
                if total_time >= max_duration:
                    break
            if not samples_list:
                return None, 0.0
            return np.concatenate(samples_list), total_time
    except Exception:
        return None, 0.0

def detect_audio_music(video_path: str, max_duration: float = 30.0) -> Tuple[bool, float, str]:
    """
    Phân tích âm thanh của video để xác định có nhạc nền (BGM) hay không nhạc (Im lặng / Cắt tiếng / Chỉ tiếng ồn).
    Trả về: (has_music, music_score, detail_desc)
    """
    if not check_has_audio(video_path):
        return False, 0.0, "Không có luồng âm thanh"

    audio, dur = extract_audio_samples(video_path, max_duration=max_duration)
    if audio is None or len(audio) < 8000:
        return False, 0.0, "Không có dữ liệu âm thanh"

    try:
        sr = 16000
        # 1. Kiểm tra năng lượng RMS
        rms = np.sqrt(np.mean(audio**2))
        db = 20 * np.log10(rms + 1e-9)
        if db < -42.0:
            return False, 0.0, f"Âm thanh im lặng ({db:.1f} dB)"

        # 2. Windowed RMS continuity (Độ liên tục âm thanh)
        win_len = int(sr * 0.5)
        hop = int(sr * 0.25)
        num_wins = (len(audio) - win_len) // hop
        if num_wins <= 0:
            return False, 0.0, "Thời lượng âm thanh quá ngắn"

        win_rms = np.array([np.sqrt(np.mean(audio[i*hop : i*hop + win_len]**2)) for i in range(num_wins)])
        active_ratio = float(np.mean(win_rms > 0.008))

        # 3. Phổ tần số & Độ phẳng phổ (Spectral Flatness)
        freqs, times, Sxx = signal.spectrogram(audio, sr, nperseg=1024, noverlap=512)
        Sxx = np.maximum(Sxx, 1e-10)
        geo_mean = np.exp(np.mean(np.log(Sxx), axis=0))
        arith_mean = np.mean(Sxx, axis=0)
        flatness = float(np.median(geo_mean / (arith_mean + 1e-9)))

        # 4. Tính đỉnh hài phổ (Harmonic Peakiness)
        peakiness = float(np.median(np.max(Sxx, axis=0) / (np.median(Sxx, axis=0) + 1e-9)))

        score = 0.0
        if flatness < 0.12: score += 40.0
        elif flatness < 0.22: score += 20.0

        if peakiness > 20.0: score += 35.0
        elif peakiness > 10.0: score += 15.0

        if active_ratio > 0.60: score += 25.0
        elif active_ratio > 0.35: score += 10.0

        has_music = score >= 50.0
        detail = f"Điểm nhạc: {int(score)}% (Độ phủ: {int(active_ratio*100)}%, Độ phẳng: {flatness:.2f})"
        return has_music, score, detail

    except Exception as e:
        return False, 0.0, f"Lỗi phân tích nhạc: {e}"

def check_video_shopee_filters(
    video_path: str,
    config: Dict[str, Any]
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Kiểm tra các tiêu chuẩn lọc video mặc định:
    1. Xóa video có độ phân giải dưới 720p (ngưỡng min_res = 720)
    2. Xóa video có định dạng không phải 9:16 (video dọc chuẩn Shopee/TikTok)
    3. Xóa video có thời lượng dưới 10s (ngưỡng min_dur = 10.0s)
    4. Xóa video có hình ảnh con người xuất hiện trong video (khuôn mặt / người)

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

    delete_has_human = config.get("delete_has_human", True)
    human_samples = config.get("human_sample_frames", 6)

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

    # 4. Kiểm tra sự xuất hiện của con người (nếu các tiêu chuẩn khác đã đạt)
    if len(reasons_short) == 0 and delete_has_human:
        has_human, human_detail, _ = detect_human_in_video(video_path, num_samples=human_samples, config=config)
        if has_human:
            reasons_short.append("Có người")
            reasons_detail.append(human_detail)

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
            category=CATEGORY_UNBOX_NOMUSIC,
            category_name=CATEGORY_NAMES[CATEGORY_UNBOX_NOMUSIC],
            total_words=0,
            total_speech_time=0.0,
            speech_rate=0.0,
            detected_language="none",
            sample_text="(Không có luồng âm thanh)",
            destination_path="",
            file_size_mb=file_size_mb,
            processing_time=elapsed,
            has_music=False,
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

        # Phân loại: Review vs Unbox (Có nhạc / Không nhạc)
        is_review = False
        has_bgm = False
        unbox_desc = ""

        if tong_so_tu < unbox_max_words or tong_thoi_gian < unbox_max_dur:
            # PHÂN LOẠI UNBOX: Chia thành Có Nhạc vs Không Nhạc
            has_bgm, bgm_score, bgm_detail = detect_audio_music(video_path)
            if has_bgm:
                category = CATEGORY_UNBOX_MUSIC
                unbox_desc = f"(Có nhạc nền | {bgm_detail})"
            else:
                category = CATEGORY_UNBOX_NOMUSIC
                unbox_desc = f"(Không nhạc | {bgm_detail})"
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
        if not sample_preview:
            sample_preview = unbox_desc if category in (CATEGORY_UNBOX_MUSIC, CATEGORY_UNBOX_NOMUSIC) else "(Không có lời thoại)"
        elif len(sample_preview) > 100:
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
            sample_text=sample_preview,
            destination_path="",
            file_size_mb=file_size_mb,
            processing_time=elapsed,
            has_subtitles=has_sub,
            subtitle_sample=sub_sample,
            has_music=has_bgm,
            error_message=""
        )

    except IndexError:
        elapsed = round(time.time() - start_time, 2)
        has_bgm, bgm_score, bgm_detail = detect_audio_music(video_path)
        cat = CATEGORY_UNBOX_MUSIC if has_bgm else CATEGORY_UNBOX_NOMUSIC
        return ClassificationResult(
            file_name=file_name,
            file_path=video_path,
            category=cat,
            category_name=CATEGORY_NAMES[cat],
            total_words=0,
            total_speech_time=0.0,
            speech_rate=0.0,
            detected_language="none",
            sample_text=f"(Âm thanh rỗng / {bgm_detail})",
            destination_path="",
            file_size_mb=file_size_mb,
            processing_time=elapsed,
            has_music=has_bgm,
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
    elif result.category == CATEGORY_UNBOX_MUSIC:
        # Thư mục con Có Nhạc bên trong 1_Unbox_Khong_Voice
        base_unbox_dir = dest_folders.get(CATEGORY_UNBOX, "")
        music_folder = config.get("output_unbox_music", "1_Co_Nhac")
        target_dir = os.path.join(base_unbox_dir, music_folder)
    elif result.category == CATEGORY_UNBOX_NOMUSIC:
        # Thư mục con Không Nhạc bên trong 1_Unbox_Khong_Voice
        base_unbox_dir = dest_folders.get(CATEGORY_UNBOX, "")
        nomusic_folder = config.get("output_unbox_nomusic", "2_Khong_Nhac")
        target_dir = os.path.join(base_unbox_dir, nomusic_folder)
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
