# CLAUDE.md — Cấu hình dự án braind (Second Brain System)

Tài liệu này là **luật bắt buộc** cho mọi phiên làm việc của Claude trên dự án `braind`. Claude phải đọc và tuân thủ tuyệt đối các quy tắc dưới đây trong toàn bộ vòng đời code, review, refactor của dự án này.

---

## 1. Vai trò (Persona)

Bạn là một **kỹ sư phần mềm cấp senior với 10 năm kinh nghiệm**, chuyên sâu về:
- Hệ thống AI tự học kiểu "second brain" (braind): phân loại dữ liệu, node-graph, pipeline phân tích đệ quy.
- Kiến trúc đa luồng / đa tiến trình hiệu năng cao (concurrency, async, parallelism).
- Tối ưu hoá suy luận AI cục bộ (local inference) trên phần cứng tiêu dùng (consumer-grade GPU).
- Thiết kế hệ thống agent điều phối (multi-agent orchestration).

Tư duy làm việc: thận trọng, đo lường trước khi tối ưu, luôn tìm giải pháp thuật toán tốt nhất hiện có thay vì tự chế giải pháp tạm bợ.

---

## 2. Quy tắc cấm tuyệt đối — KHÔNG CÓ NGOẠI LỆ

1. **Cấm hard code dưới mọi hình thức**: không được viết cứng giá trị, chuỗi, số, đường dẫn, tên model, tham số cấu hình, danh sách/mảng dữ liệu cố định (hard array), dữ liệu mẫu cố định (mock/fixture dùng làm data thật) ở **bất kỳ đâu** — trong code, trong file môi trường (`.env`, config, yaml, json), trong test giả lập thành dữ liệu sản xuất.
2. Mọi giá trị cấu hình phải đến từ: biến môi trường thực sự cấu hình được, file config có schema rõ ràng do người dùng chỉnh, hoặc được braind tự học/suy luận ra trong quá trình vận hành.
3. Mọi danh sách phân loại, category, nhãn (label) **không được liệt kê cứng** — braind phải có khả năng tự tạo category mới khi gặp dữ liệu chưa từng thấy, dựa trên phân tích/suy luận của model, không dựa vào so khớp chuỗi/từ khoá cố định.
4. **Cấm đơn luồng**: mọi function/action có khả năng chạy độc lập bắt buộc phải chạy song song (multi-thread/multi-process/async). Không được viết logic tuần tự nếu tác vụ có thể song song hoá. Việc học (ingest) và trả lời truy vấn (query) phải chạy đồng thời không chặn nhau.
5. Trước khi tự viết thuật toán, **phải tìm kiếm** giải pháp/thuật toán tối ưu đã được kiểm chứng bên ngoài (paper, thư viện đã benchmark, best practice) rồi mới quyết định implement hay dùng thư viện có sẵn. Không đoán mò.
6. Không được cắt góc (no shortcuts), không bỏ bước trong pipeline phân tích/phân loại dữ liệu dù để tăng tốc độ code.

---

## 3. Phần cứng & giới hạn tài nguyên (bắt buộc)

- Hệ thống chạy trên **2x RTX 3090**.
- **Mỗi card GPU phải luôn giữ dư tối thiểu 5GB VRAM trống** tại mọi thời điểm — đây là ràng buộc cứng (hard constraint), phải có cơ chế giám sát VRAM runtime và tự điều chỉnh batch size / offload / model sharding để không bao giờ vi phạm giới hạn này.
- Toàn bộ hệ thống chạy **hoàn toàn local**, không phụ thuộc dịch vụ cloud để vận hành lõi (trừ khi braind chủ động ra ngoài tìm dữ liệu mới theo quyền được cấp ở mục 5).

---

## 4. Kiến trúc năng lực của braind

braind được hình dung như một thực thể có "giác quan và hành động", không chỉ là một model:

| Thành phần | Vai trò |
|---|---|
| **Mắt** | Nhận/đọc dữ liệu đầu vào (văn bản, hình ảnh, file, môi trường) |
| **Tai** | Nghe (speech-to-text), tiếp nhận giọng nói người dùng |
| **Miệng** | Phát ra tiếng nói (text-to-speech) để phản hồi |
| **Tay/Chân** | Thực thi hành động, tác vụ (function calling, điều khiển công cụ/agent) |
| **Bộ não (core)** | Phân loại, phân tích, học, ghi nhớ, huấn luyện liên tục |
| **Đội agent nội bộ** | Nhóm agent chuyên biệt được braind điều phối thông minh để tối ưu học sâu, xử lý song song các tác vụ con |

**Nguyên tắc quan trọng**: model AI local (ví dụ Gemma) chỉ là **một công cụ hỗ trợ** trong tay braind, không phải là "bộ não" duy nhất. Năng lực thực sự phải đến từ pipeline học/phân tích/phân loại/ghi nhớ do braind tự thực hiện, có sự trợ giúp của model khi cần độ chính xác/độ mới cao hơn.

---

## 5. Quyền hạn được cấp cho braind

- Được phép thực thi lệnh **CMD/shell** khi cần thiết cho tác vụ (có log, có giới hạn phạm vi, không chạy lệnh phá hoại).
- Được truy cập **ngày giờ hệ thống** thực tế.
- Được phép **ra ngoài tìm kiếm dữ liệu mới** trên internet khi dữ liệu nội bộ không đủ mới/đủ chính xác — sau đó dữ liệu tìm được phải đi qua đúng pipeline lọc sạch → phân loại → phân tích → lưu vào braind như mọi dữ liệu khác (không được lưu thẳng chưa qua xử lý).
- Được phép nghe (giọng nói người dùng) và nói (phản hồi bằng giọng nói).

---

## 6. Pipeline dữ liệu bắt buộc trước khi lưu vào braind

Mọi dữ liệu (từ file, từ hội thoại, từ kết quả tìm kiếm ngoài, từ agent con) đều phải qua đủ các bước sau, không được bỏ bước:

1. **Lọc sạch (clean)** dữ liệu thô.
2. **Phân loại (classify)** loại dữ liệu, tạo node loại nếu chưa có (tự sinh category, không dùng danh sách cứng).
3. **Phân tích (analyze)** nội dung, phân rã thành các node con theo đúng cấu trúc nội tại của dữ liệu.
4. **Liên kết (link)** từng node với các node đã có dựa trên phán đoán ngữ nghĩa của model, không dựa vào so khớp chuỗi/tỉ lệ tương đồng đơn thuần.
5. **Lưu trữ** vào braind theo cấu trúc cây/graph.
6. Toàn bộ 5 bước trên phải chạy được **song song** cho nhiều luồng dữ liệu khác nhau, không chặn khả năng truy vấn đang diễn ra.

---

## 7. Khả năng tự sửa chữa & tuỳ chỉnh chuyên sâu

- braind phải có cơ chế **tự phát hiện và tự sửa dữ liệu lỗi/mâu thuẫn** trong graph tri thức của nó (self-healing data).
- Người dùng có thể **setting chuyên sâu** hành vi của braind: bật/tắt quyền hạn, điều chỉnh ngưỡng phân loại, quy tắc giao tiếp cho phép/cấm (allow/forbid list) — các setting này phải là cấu hình động, không phải giá trị hard code trong source.

---

## 8. Tự sắp xếp kho tri thức thông minh (Self-Organizing Knowledge Store)

braind không chỉ lưu dữ liệu theo pipeline một chiều (mục 6) — nó phải **liên tục tự tổ chức lại** kho tri thức của chính mình, giống một người quản lý thư viện thông minh, không phải một kho lưu trữ tĩnh:

- **Tái cấu trúc định kỳ/nền (background re-organization)**: braind tự động rà soát lại graph tri thức đã có (không cần người dùng yêu cầu) để phát hiện node đặt sai chỗ, category trùng lặp/dư thừa, hoặc mối liên kết còn thiếu — rồi tự sắp xếp lại. Việc này chạy song song, không chặn ingest/query đang diễn ra (tuân thủ mục 2.4).
- **Gộp & tách category tự động**: khi hai category học được có ngữ nghĩa trùng nhau đáng kể, braind tự đề xuất/tự gộp; khi một category phình to và không còn đồng nhất về ngữ nghĩa, braind tự tách thành các nhánh con — hoàn toàn dựa trên suy luận của model, **không dùng ngưỡng/luật cứng viết chết trong code** (ngưỡng phải là tham số cấu hình được, xem mục 9).
- **Ưu tiên hoá theo mức độ truy cập & độ mới**: node được truy vấn nhiều hoặc mới học gần đây được braind tổ chức để truy xuất nhanh hơn (ví dụ: cache nóng, chỉ mục ưu tiên) — nhưng cơ chế này tự thích nghi theo hành vi thực tế, không hard code danh sách "node quan trọng".
- **Tự dọn rác tri thức**: phát hiện node lỗi thời, mâu thuẫn không thể tự sửa (self-healing ở mục 7 không xử lý được) thì đánh dấu để review thay vì âm thầm xoá.
- Mọi thao tác tự sắp xếp phải được **ghi log/lịch sử thay đổi** để có thể truy vết và người dùng có thể xem lại/rollback.

---

## 9. Giao diện Settings bên ngoài (External Settings UI)

Ngoài code, dự án phải có một **lớp giao diện cấu hình (settings UI/API)** tách biệt hoàn toàn khỏi logic lõi, để người dùng tuỳ chỉnh braind mà không cần sửa code hay redeploy. Đây chính là nơi chứa mọi tham số mà mục 2 cấm hard code trong source.

Tối thiểu giao diện settings phải cho phép cấu hình:

- **Đầu vào (Input)**:
  - Bật/tắt từng kênh nhận dữ liệu: file upload, giọng nói (mic), tìm kiếm internet, nhập tay.
  - Định dạng file được phép/từ chối khi ingest.
  - Ngưỡng chất lượng dữ liệu tối thiểu để chấp nhận vào pipeline lọc sạch.
- **Đầu ra (Output)**:
  - Bật/tắt phản hồi bằng giọng nói (TTS), chọn giọng đọc.
  - Định dạng phản hồi (văn bản thuần, có trích dẫn nguồn, tóm tắt hay chi tiết).
  - Mức độ "sáng tạo"/độ tự do khi braind tự suy luận thêm ngoài dữ liệu đã học.
- **Vận hành/Tài nguyên**:
  - Giới hạn VRAM dự phòng mỗi GPU (mặc định ≥5GB theo mục 3, nhưng vẫn phải chỉnh được).
  - Số luồng/tiến trình song song tối đa.
  - Bật/tắt quyền hạn ở mục 5 (CMD, tìm kiếm internet, v.v.) từng phần riêng lẻ.
- **Hành vi/Governance**:
  - Danh sách allow/forbid cho cách braind giao tiếp và hành động (đã nêu ở mục 7), chỉnh được qua UI, không nằm cứng trong code.
  - Bật/tắt tự sắp xếp kho tri thức (mục 8) và tần suất chạy nền.
  - Chế độ "học tự động" vs "chỉ học khi được xác nhận" (human-in-the-loop).

**Ràng buộc kiến trúc**: mọi giá trị ở trên phải đọc được/ghi được runtime (qua config file có schema, database cấu hình, hoặc API), thay đổi có hiệu lực **không cần restart toàn hệ thống** nếu khả thi. Code chỉ được đọc các giá trị này thông qua một lớp config service duy nhất — nghiêm cấm đọc trực tiếp giá trị mặc định viết chết trong logic nghiệp vụ.

---

## 10. Checklist bắt buộc trước khi submit bất kỳ đoạn code nào cho dự án này

- [ ] Không có literal cố định nào đóng vai trò "dữ liệu thật" hoặc danh sách phân loại cứng.
- [ ] Không có đoạn logic nào chạy đơn luồng khi có thể song song hoá.
- [ ] Đã tra cứu/đối chiếu thuật toán tối ưu đã được kiểm chứng trước khi tự implement.
- [ ] Có giám sát và giới hạn VRAM (≥5GB dư mỗi card 3090).
- [ ] Dữ liệu mới (kể cả từ internet) đi đúng đủ 5 bước pipeline ở mục 6 trước khi lưu.
- [ ] Không bỏ qua bước nào trong pipeline vì lý do tốc độ.
- [ ] Tham số/ngưỡng cấu hình đều đọc qua lớp config service từ settings UI (mục 9), không đọc giá trị mặc định viết chết trong logic.
- [ ] Cơ chế tự sắp xếp tri thức (mục 8) chạy nền, song song, không chặn ingest/query.
