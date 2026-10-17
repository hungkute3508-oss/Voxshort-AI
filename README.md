# 🎬 VoxSort AI - Công Cụ Phân Loại Video Tự Động (Whisper Voice AI & RapidOCR)

**VoxSort AI** là phần mềm desktop chuyên nghiệp được xây dựng để quét và tự động phân loại khối lượng lớn video (Unbox, Review, TikTok, Reels, Shorts, Youtube...) dựa trên trí tuệ nhân tạo nhận diện giọng nói **Faster-Whisper (OpenAI)** kết hợp thị giác máy tính **RapidOCR (ONNX)** nhận diện phụ đề màn hình.

---

## ✨ Cấu Trúc Phân Loại Tự Động

Tool tự động phân loại kho video thành các nhóm rõ ràng:

1. **📦 UNBOX (Không Voice / Nhạc nền)**:
   - Video không có tiếng nói hoặc chỉ có nhạc nền / hiệu ứng âm thanh ngắt quãng ngắn (< 8 từ hoặc < 2.0 giây).
   - Tự động phát hiện video không có track âm thanh trong 0.01s.
   - Thư mục đích: `1_Unbox_Khong_Voice`

2. **🎙️ REVIEW (Có Voice Thuyết Minh)**:
   - Video có người nói liên tục (≥ 15 từ và tốc độ nói ≥ 1.5 từ/giây).
   - **Tự động chia thành 2 nhóm con bằng RapidOCR**:
     - 🟢 **`2_Review_Co_Voice\1_Co_Phu_De`**: Video có phụ đề / captions xuất hiện trên khung hình theo dòng nói.
     - 🔵 **`2_Review_Co_Voice\2_Khong_Phu_De`**: Video review thuần túy bằng giọng nói, không chèn chữ phụ đề trên video.

3. **⚠️ CẦN KIỂM TRA LẠI (Nghi Ngờ)**:
   - Các video nằm giữa ranh giới (ví dụ nói vài từ ngắn rồi im lặng) để người dùng xem lại thủ công.
   - Thư mục đích: `3_Can_Kiem_Tra_Lai`

4. **🗑️ BỘ LỌC TỰ ĐỘNG XÓA (MẶC ĐỊNH - CHUẨN SHOPEE VIDEO)**:
   - **Xóa video có độ phân giải dưới 720p**: Video có cạnh ngắn < 720px (như 360p, 480p, 540p) sẽ bị tự động xóa.
   - **Xóa video không phải tỉ lệ 9:16**: Chỉ giữ video dọc chuẩn Shopee/TikTok 9:16 (ví dụ: 1080x1920, 720x1280). Tự động xóa video ngang 16:9, vuông 1:1, 3:4,...
   - **Xóa video có thời lượng dưới 10 giây**: Video < 10.0s sẽ tự động bị xóa.
   - ⚡ **Siêu tốc & Tiết kiệm tài nguyên**: Kiểm tra thông số chỉ mất 0.01s - 0.02s, xóa trực tiếp khỏi ổ đĩa ngay trước khi gọi Whisper AI, giúp tăng tốc toàn bộ quá trình lên gấp nhiều lần.

---

## 🔍 Thuật Toán Nhận Diện Phụ Đề Hoạt Động Như Thế Nào?

Để nhận diện phụ đề **chuẩn xác và không làm chậm máy**, VoxSort AI áp dụng cơ chế:
* **Chỉ quét khi đã xác nhận là Review**: Không tốn tài nguyên quét OCR trên video Unbox.
* **Lấy mẫu thông minh (Sampling)**: Lấy mẫu 6 khung hình trải đều từ 15% đến 85% thời lượng video.
* **Khoanh vùng phụ đề (ROI)**: Chỉ quét vùng từ 35% đến 92% chiều cao màn hình (bỏ qua header, tên tài khoản Instagram/TikTok ở trên cùng và thanh điều hướng ở đáy).
* **Loại trừ Watermark & Logo cố định**: Phụ đề thực sự sẽ có text thay đổi liên tục theo thời gian; nếu text ở mọi frame giống hệt nhau (logo kênh), tool sẽ không tính là phụ đề.
* **Tốc độ cực nhanh**: Toàn bộ quá trình quét OCR chỉ mất khoảng **0.15s – 0.25s / video** nhờ ONNX Runtime!

---

## 🚀 Hướng Dẫn Sử Dụng

### Cách 1: Chạy 1-Click (Khuyên Dùng)
Nhấp đúp chuột vào file:
```
Chay_VoxSort.bat
```

### Cách 2: Chạy Bằng Dòng Lệnh Terminal
```bash
python main.py
```

---

## 📁 Cấu Trúc Thư Mục Kết Quả

Sau khi quét, ngay bên trong thư mục video gốc sẽ xuất hiện:
```text
Thư_Mục_Gốc/
├── 1_Unbox_Khong_Voice/
├── 2_Review_Co_Voice/
│   ├── 1_Co_Phu_De/        <-- Video Review có chữ phụ đề trên màn hình
│   └── 2_Khong_Phu_De/     <-- Video Review không có phụ đề
└── 3_Can_Kiem_Tra_Lai/
```
