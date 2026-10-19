# 🎬 VoxSort AI - Công Cụ Phân Loại Video Tự Động (Whisper Voice AI & RapidOCR)

**VoxSort AI** là phần mềm desktop chuyên nghiệp được xây dựng để quét và tự động phân loại khối lượng lớn video (Unbox, Review, TikTok, Reels, Shorts, Youtube...) dựa trên trí tuệ nhân tạo nhận diện giọng nói **Faster-Whisper (OpenAI)** kết hợp thị giác máy tính **RapidOCR (ONNX)** nhận diện phụ đề màn hình.

---

## ✨ Cấu Trúc Phân Loại Tự Động

Tool tự động phân loại kho video thành các nhóm rõ ràng:

1. **📦 UNBOX (Không Voice / Video Sản Phẩm)**:
   - Video không có tiếng nói thuyết minh hoặc chỉ có hiệu ứng âm thanh ngắt quãng ngắn (< 8 từ hoặc < 2.0 giây).
   - **Tự động chia thành 2 nhóm con**:
     - 🎵 **`1_Unbox_Khong_Voice\1_Co_Nhac`**: Video có nhạc nền (BGM / bài hát / giai điệu trending).
     - 🔇 **`1_Unbox_Khong_Voice\2_Khong_Nhac`**: Video không có nhạc (im lặng, tắt âm hoặc chỉ có tiếng động môi trường/unbox).

2. **🎙️ REVIEW (Có Voice Thuyết Minh)**:
   - Video có người nói liên tục (≥ 15 từ và tốc độ nói ≥ 1.5 từ/giây).
   - **Tự động chia thành 2 nhóm con bằng RapidOCR**:
     - 🟢 **`2_Review_Co_Voice\1_Co_Phu_De`**: Video có phụ đề / captions xuất hiện trên khung hình theo dòng nói.
     - 🔵 **`2_Review_Co_Voice\2_Khong_Phu_De`**: Video review thuần túy bằng giọng nói, không chèn chữ phụ đề trên video.

3. **⚠️ CẦN KIỂM TRA LẠI (Nghi Ngờ)**:
   - Các video nằm giữa ranh giới (ví dụ nói vài từ ngắn rồi im lặng) để người dùng xem lại thủ công.
   - Thư mục đích: `3_Can_Kiem_Tra_Lai`

4. **🗑️ BỘ LỌC TỰ ĐỘNG XÓA (MẶC ĐỊNH - CHUẨN SHOPEE VIDEO)**:
   - **Xóa video xuất hiện cả người hoặc có mặt người**: Tự động quét AI (YuNet Face Detector & MediaPipe Person Detector) nhận diện khuôn mặt người và dáng cả người (toàn thân / nửa thân).
     - 💡 **Quy tắc thông minh**: Nếu chỉ có **cánh tay / bàn tay** cầm nắm sản phẩm, mở hộp (unbox), swatch hoặc chỉ ngón tay giới thiệu sản phẩm thì **GIỮ LẠI** và tiến hành lọc bình thường (không bị xóa)!
     - Chỉ xóa khi nhân vật xuất hiện cả người (reviewer/người mẫu) hoặc để lộ khuôn mặt trên khung hình.
   - **Xóa video có độ phân giải dưới 720p**: Video có cạnh ngắn < 720px (như 360p, 480p, 540p) sẽ bị tự động xóa.
   - **Xóa video không phải tỉ lệ 9:16**: Chỉ giữ video dọc chuẩn Shopee/TikTok 9:16 (ví dụ: 1080x1920, 720x1280). Tự động xóa video ngang 16:9, vuông 1:1, 3:4,...
   - **Xóa video có thời lượng dưới 10 giây**: Video < 10.0s sẽ tự động bị xóa.
   - ⚡ **Siêu tốc & Tiết kiệm tài nguyên**: Kiểm tra thông số chỉ mất 0.01s - 0.05s, xóa trực tiếp khỏi ổ đĩa ngay trước khi gọi Whisper AI, giúp tăng tốc toàn bộ quá trình lên gấp nhiều lần.

---

## 🔍 Thuật Toán Nhận Diện Phụ Đề & Nhận Diện Người Hoạt Động Như Thế Nào?

* **Nhận diện con người (Human & Face vs Hands)**: Kết hợp YuNet Face Detector và MediaPipe Person Detector:
  - *YuNet*: Kiểm tra 5 điểm mốc giải phẫu khuôn mặt (2 mắt, mũi, 2 khóe miệng) và kích thước tối thiểu, loại bỏ ảnh vẽ nhỏ trên nhãn sản phẩm hoặc vân tay phản chiếu.
  - *MediaPipe PersonDet*: Chỉ kích hoạt khi phát hiện toàn thân hoặc nửa thân trên (vai, ngực, lưng) chiếm từ 18% chiều cao màn hình trở lên. Đôi tay hoặc cánh tay cầm nắm sản phẩm không kích hoạt dáng người nên luôn được giữ lại an toàn.
* **Nhận diện nhạc nền (Music Detection)**: Phân tích phổ âm thanh (Spectral Flatness, Harmonic Peakiness, Active Energy Continuity) để phân biệt chính xác giữa nhạc nền giai điệu và tiếng sột soạt/im lặng của unbox.
* **Nhận diện phụ đề (OCR Subtitles)**: Lấy mẫu khung hình và khoanh vùng phụ đề (ROI 35% - 92% chiều cao), loại trừ logo/watermark cố định.

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
│   ├── 1_Co_Nhac/          <-- Video Unbox có nhạc nền
│   └── 2_Khong_Nhac/       <-- Video Unbox không có nhạc (im lặng / chỉ có tiếng unbox)
├── 2_Review_Co_Voice/
│   ├── 1_Co_Phu_De/        <-- Video Review có chữ phụ đề trên màn hình
│   └── 2_Khong_Phu_De/     <-- Video Review không có phụ đề
└── 3_Can_Kiem_Tra_Lai/
```
