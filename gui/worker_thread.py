import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from PyQt6.QtCore import QThread, pyqtSignal

from classifier_core import (
    VideoInfo, ClassificationResult, transcribe_and_classify,
    dispatch_file_action, CATEGORY_UNBOX, CATEGORY_UNBOX_MUSIC, CATEGORY_UNBOX_NOMUSIC,
    CATEGORY_REVIEW, CATEGORY_REVIEW_SUB, CATEGORY_REVIEW_NOSUB,
    CATEGORY_DOUBT, CATEGORY_ERROR, CATEGORY_DELETED, get_ocr_instance,
    check_video_shopee_filters, delete_video_file
)

class ClassifierWorker(QThread):
    """Worker chạy ngầm phân tích và phân loại danh sách video."""

    # Signals
    sig_log = pyqtSignal(str, str)  # (message, level: "info" | "success" | "warn" | "error")
    sig_progress = pyqtSignal(int, int, str, float, float)  # (current, total, current_file, elapsed_sec, eta_sec)
    sig_result = pyqtSignal(object)  # ClassificationResult
    sig_finished = pyqtSignal(int, dict)  # (total_done, summary_stats)
    sig_model_loaded = pyqtSignal(str)

    def __init__(self, videos: List[VideoInfo], config: Dict[str, Any], dest_folders: Dict[str, str]):
        super().__init__()
        self.videos = videos
        self.config = config
        self.dest_folders = dest_folders

        self._is_paused = False
        self._is_stopped = False
        self._pause_cond = threading.Condition()

    def pause(self):
        with self._pause_cond:
            self._is_paused = True
        self.sig_log.emit("⏸️ Đã tạm dừng quá trình phân loại.", "warn")

    def resume(self):
        with self._pause_cond:
            self._is_paused = False
            self._pause_cond.notify_all()
        self.sig_log.emit("▶️ Tiếp tục quá trình phân loại.", "info")

    def stop(self):
        with self._pause_cond:
            self._is_stopped = True
            self._is_paused = False
            self._pause_cond.notify_all()
        self.sig_log.emit("⏹️ Đang yêu cầu dừng tiến trình...", "warn")

    def is_paused(self) -> bool:
        return self._is_paused

    def is_stopped(self) -> bool:
        return self._is_stopped

    def run(self):
        total_videos = len(self.videos)
        if total_videos == 0:
            self.sig_finished.emit(0, {})
            return

        model_name = self.config.get("model_name", "tiny")
        device = self.config.get("device", "cpu")
        compute_type = self.config.get("compute_type", "int8")
        workers_count = max(1, min(self.config.get("workers", 2), 16))
        split_sub = self.config.get("split_review_subtitles", False)

        sub_info = " (Bật OCR nhận diện Phụ đề)" if split_sub else ""
        self.sig_log.emit(
            f"🚀 Khởi động Whisper AI (Model: '{model_name}', Device: {device.upper()}, Compute: {compute_type}, Luồng: {workers_count}){sub_info}...",
            "info"
        )

        # Thông báo trạng thái các tiêu chuẩn lọc Shopee
        del_low_res = self.config.get("delete_low_res", True)
        del_non_9_16 = self.config.get("delete_non_9_16", True)
        del_short = self.config.get("delete_short_duration", True)
        filter_status = []
        if del_low_res: filter_status.append(f"< {self.config.get('filter_min_resolution', 720)}p")
        if del_non_9_16: filter_status.append("!= 9:16")
        if del_short: filter_status.append(f"< {int(self.config.get('filter_min_duration', 10))}s")
        if filter_status:
            self.sig_log.emit(f"🧹 Tiêu chuẩn tự động xóa đang BẬT: [Xóa: {', '.join(filter_status)}]", "info")

        # 1. Khởi tạo Faster-Whisper Model
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                num_workers=workers_count
            )
            self.sig_model_loaded.emit(model_name)
            self.sig_log.emit(f"✅ Mô hình Whisper [{model_name}] đã sẵn sàng!", "success")
        except Exception as e:
            self.sig_log.emit(f"❌ Không thể tải mô hình Whisper: {e}", "error")
            self.sig_finished.emit(0, {"error": str(e)})
            return

        # 2. Khởi động trước RapidOCR nếu bật tách phụ đề
        if split_sub:
            self.sig_log.emit("🔍 Đang tải trước bộ máy RapidOCR nhận diện phụ đề...", "info")
            ocr = get_ocr_instance()
            if ocr:
                self.sig_log.emit("✅ RapidOCR AI đã sẵn sàng quét phụ đề!", "success")

        start_time = time.time()
        processed_count = 0
        summary = {
            CATEGORY_UNBOX: 0,
            CATEGORY_UNBOX_MUSIC: 0,
            CATEGORY_UNBOX_NOMUSIC: 0,
            CATEGORY_REVIEW: 0,
            CATEGORY_REVIEW_SUB: 0,
            CATEGORY_REVIEW_NOSUB: 0,
            CATEGORY_DOUBT: 0,
            CATEGORY_ERROR: 0,
            CATEGORY_DELETED: 0
        }

        # Hàm xử lý từng video
        def _process_one(v: VideoInfo) -> ClassificationResult:
            if self._is_stopped:
                return None

            with self._pause_cond:
                while self._is_paused and not self._is_stopped:
                    self._pause_cond.wait(0.2)

            if self._is_stopped:
                return None

            # BƯỚC 1: KIỂM TRA BỘ LỌC SHOPEE (< 720p, != 9:16, < 10s)
            should_del, r_short, r_detail, specs = check_video_shopee_filters(v.file_path, self.config)
            if should_del:
                del_ok, del_err = delete_video_file(v.file_path)
                short_reasons_str = ", ".join(r_short)
                detail_reasons_str = " | ".join(r_detail)

                return ClassificationResult(
                    file_name=v.file_name,
                    file_path=v.file_path,
                    category=CATEGORY_DELETED,
                    category_name=f"🗑️ Đã xóa ({short_reasons_str})",
                    total_words=0,
                    total_speech_time=0.0,
                    speech_rate=0.0,
                    detected_language="-",
                    sample_text=f"[ĐÃ XÓA] {detail_reasons_str}" if del_ok else f"[LỖI XÓA] {del_err}",
                    destination_path="(Đã xóa khỏi ổ đĩa)" if del_ok else v.file_path,
                    file_size_mb=v.file_size_mb,
                    processing_time=specs.get("check_time", 0.02),
                    has_subtitles=False,
                    subtitle_sample="",
                    error_message=del_err if not del_ok else "",
                    resolution=f"{specs.get('width', 0)}x{specs.get('height', 0)}",
                    aspect_ratio=specs.get("ratio_str", "-"),
                    video_duration=specs.get("duration", 0.0)
                )

            # BƯỚC 2: Phân tích âm thanh Whisper + OCR phụ đề cho video đạt chuẩn
            res = transcribe_and_classify(model, v.file_path, self.config)
            res.resolution = f"{specs.get('width', 0)}x{specs.get('height', 0)}"
            res.aspect_ratio = specs.get("ratio_str", "-")
            res.video_duration = specs.get("duration", 0.0)

            # BƯỚC 3: Thực hiện di chuyển hoặc copy file
            if not self._is_stopped:
                dispatch_file_action(res, self.config, self.dest_folders)

            return res

        # Chạy song song đa luồng
        with ThreadPoolExecutor(max_workers=workers_count) as executor:
            future_to_video = {executor.submit(_process_one, v): v for v in self.videos}

            for future in as_completed(future_to_video):
                if self._is_stopped:
                    break

                try:
                    result: ClassificationResult = future.result()
                    if not result:
                        continue

                    processed_count += 1
                    summary[result.category] = summary.get(result.category, 0) + 1
                    if result.category in (CATEGORY_REVIEW_SUB, CATEGORY_REVIEW_NOSUB):
                        summary[CATEGORY_REVIEW] = summary.get(CATEGORY_REVIEW, 0) + 1
                    if result.category in (CATEGORY_UNBOX_MUSIC, CATEGORY_UNBOX_NOMUSIC):
                        summary[CATEGORY_UNBOX] = summary.get(CATEGORY_UNBOX, 0) + 1

                    elapsed = time.time() - start_time
                    avg_per_item = elapsed / processed_count if processed_count > 0 else 0
                    remaining_items = total_videos - processed_count
                    eta = avg_per_item * remaining_items

                    # Bắn signal kết quả cho UI
                    self.sig_result.emit(result)
                    self.sig_progress.emit(processed_count, total_videos, result.file_name, elapsed, eta)

                    # Log trạng thái
                    if result.category == CATEGORY_DELETED:
                        self.sig_log.emit(
                            f"[{processed_count}/{total_videos}] {result.file_name} -> {result.category_name} "
                            f"(Độ phân giải: {result.resolution}, Tỉ lệ: {result.aspect_ratio}, Thời lượng: {result.video_duration}s)",
                            "warn"
                        )
                    else:
                        log_type = "info"
                        extra_sub = ""
                        if result.category in (CATEGORY_REVIEW, CATEGORY_REVIEW_SUB):
                            log_type = "success"
                            extra_sub = f" [Sub: {result.subtitle_sample}]" if result.subtitle_sample else ""
                        elif result.category == CATEGORY_REVIEW_NOSUB:
                            log_type = "info"
                            extra_sub = " [Không có phụ đề]"
                        elif result.category in (CATEGORY_UNBOX, CATEGORY_UNBOX_MUSIC, CATEGORY_UNBOX_NOMUSIC):
                            log_type = "info"
                        elif result.category == CATEGORY_DOUBT:
                            log_type = "warn"
                        else:
                            log_type = "error"

                        self.sig_log.emit(
                            f"[{processed_count}/{total_videos}] {result.file_name} -> {result.category_name}{extra_sub} "
                            f"(Từ: {result.total_words}, Thời lượng voice: {result.total_speech_time}s, Tốc độ: {result.speech_rate} t/s)",
                            log_type
                        )

                except Exception as e:
                    self.sig_log.emit(f"Lỗi khi xử lý file: {e}", "error")

        total_elapsed = time.time() - start_time
        summary["total_time"] = round(total_elapsed, 1)

        if self._is_stopped:
            self.sig_log.emit(f"⏹️ Đã dừng tiến trình. Đã xử lý {processed_count}/{total_videos} video.", "warn")
        else:
            self.sig_log.emit(
                f"🎉 Hoàn tất toàn bộ {processed_count} video trong {round(total_elapsed, 1)}s!",
                "success"
            )

        self.sig_finished.emit(processed_count, summary)
