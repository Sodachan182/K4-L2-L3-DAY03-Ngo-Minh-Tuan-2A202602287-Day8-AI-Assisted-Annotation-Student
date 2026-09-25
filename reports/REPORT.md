# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Ngô Minh Tuấn

Công cụ gán nhãn đã dùng: CVAT

## 1. Dữ liệu và cách chia tập

Pool chưa gán nhãn và tập test được chia theo trục thời gian, có vùng đệm ở giữa, để giảm khả năng cùng một cảnh hoặc các frame gần như trùng nhau xuất hiện ở cả hai tập. Video đường cao tốc ban đêm thay đổi liên tục nhưng chậm giữa các frame liền kề; vị trí xe, góc máy, ánh sáng và nền có thể gần như giống hệt nhau. Nếu chia ngẫu nhiên, near-duplicate dễ rơi vào cả train và test, khiến mô hình được đánh giá trên cảnh đã thấy gần như nguyên dạng và làm chỉ số có xu hướng lạc quan hơn khả năng tổng quát hóa thực tế.

Tập test hiện có 20 ảnh với 403 box tham chiếu; 14 box cao dưới 16 px được bỏ qua khi tính điểm. Việc tách theo thời gian giúp phép so sánh nghiêm ngặt hơn, nhưng quy mô test nhỏ vẫn làm kết quả nhạy với một số ít frame.

## 2. Mô hình khởi đầu lạnh (cold start)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Theo `metrics_round0.json`, ở ngưỡng confidence 0.25 mô hình cold start đạt AP50 0.7714, precision 0.9249, recall 0.4888 và F1 0.6396, với TP = 197, FP = 16, FN = 206. Precision cao nhưng recall thấp cho thấy vấn đề chính là bỏ sót. Recall theo kích thước là 0.1818 cho xe nhỏ (66 box tham chiếu), 0.5473 cho xe vừa (296 box) và 0.5610 cho xe lớn (41 box); xe nhỏ/xa là nhóm yếu rõ nhất.

`compare_round0.jpg` cho thấy nhiều box không khớp tập tham chiếu tập trung ở các xe nhỏ, xa, thiếu sáng hoặc chỉ hiện qua cụm đèn; đây là những trường hợp dễ bị bỏ sót hoặc khó đặt box chính xác. Ví dụ trên bốn frame minh họa, kết quả lần lượt là `frame_0050`: TP 11, FP 2, FN 7; `frame_0150`: TP 10, FP 2, FN 10; `frame_0250`: TP 6, FP 2, FN 9; và `frame_0350`: TP 9, FP 2, FN 14. Trước khi coi mọi khác biệt là lỗi mô hình, người gán nhãn cần rà lại các box tham chiếu chỉ bao quanh một vùng sáng hoặc hai điểm đèn rất xa: nếu không đủ dấu hiệu nhận ra phương tiện bốn bánh, chính nhãn tham chiếu cũng có thể chưa chắc chắn.

## 3. Chiến lược chọn mẫu

Điểm chọn mẫu có dạng `score = W_U·U + W_A·A + W_D·D`: mỗi thành phần được nhân với trọng số để cân bằng độ bất định `U`, mức mơ hồ/thông tin cần rà `A` và độ đa dạng `D`. Uncertainty nghĩa là mô hình không tự tin vào dự đoán của mình; nó hữu ích để tìm lỗi tiềm năng nhưng không chứng minh ảnh đó sẽ làm mô hình tốt hơn. `MIN_GAP_S` đặt khoảng cách thời gian tối thiểu giữa các ảnh được chọn, nhằm hạn chế lấy nhiều frame liền nhau gần trùng cảnh.

Trong `SELECTION.md`, `frame_0182.jpg` đứng hạng 1 (điểm 0.9591, `U = 0.9182`, 28 box), nên vừa có tín hiệu bất định cao vừa có chi phí rà nhãn thấp hơn nhiều ảnh đông xe. `frame_0369.jpg` đứng hạng 2 (0.9324, `U = 0.9315`, 43 box) đại diện đoạn muộn của video nhưng có chi phí cao. `frame_0326.jpg` đứng hạng 4 (0.9155, 39 box) được ưu tiên hơn `frame_0331.jpg` ở hạng 5 (0.9154, 47 box) vì điểm gần như bằng nhau, thời gian gần nhau nhưng ít box hơn. Một đối chứng khác là `frame_0380.jpg`: dù đứng hạng 3 với điểm 0.9170, ảnh này không nhất thiết nên nằm trong ngân sách năm ảnh nếu đã chọn `frame_0369.jpg`, vì cả hai thuộc đoạn muộn và contact sheet cho thấy cảnh quan tương tự. Vì vậy, uncertainty cần được cân bằng với near-duplicate, độ đa dạng cảnh và công gán nhãn.

## 4. Các vòng học chủ động (active learning)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 329 | 0.489 | -0.282 | 1.000 | 0.199 | 0.331 | 0.000 | 0.193 | 0.561 |

Ở vòng 1, mô hình đề xuất ban đầu 169 box trên 12 ảnh. Kết quả review khớp `round1_diff.json`: accepted 147, edited 8, deleted 14 và added 174; tổng nhãn sau sửa là 329 box. Số box thêm mới còn nhiều hơn toàn bộ pre-label ban đầu, cho thấy AI pre-label bỏ sót rất nhiều xe. `REVIEW_LOG.csv` ghi pattern chính là xe xa, xe mờ, xe trong vùng tối hoặc bị che một phần. Log cũng ghi một false positive cụ thể ở `frame_0331.jpg`: AI gán box vào vùng ánh sáng/phản chiếu trên mặt đường như một `car`, và box đã bị xóa vì không đủ căn cứ nhận diện phương tiện bốn bánh.

Sau fine-tune 12 ảnh trong 50 epoch, AP50 giảm từ 0.7714 xuống 0.4895, tức giảm 0.2819. So với vòng 0, TP giảm từ 197 xuống 80, FP giảm từ 16 xuống 0 và FN tăng từ 206 lên 323. Precision tăng từ 0.9249 lên 1.0000, nhưng recall giảm từ 0.4888 xuống 0.1985 và F1 giảm từ 0.6396 xuống 0.3313. Đây không phải cải thiện tổng thể: việc không tạo FP ở ngưỡng 0.25 đi kèm với việc phát hiện ít xe hơn nhiều.

Theo kích thước, recall xe nhỏ giảm từ 0.1818 xuống 0.0000; xe vừa giảm từ 0.5473 xuống 0.1926; xe lớn giữ nguyên 0.5610. `compare_round1.jpg` thể hiện cùng xu hướng. Chẳng hạn, trên `frame_0050`, cold start có TP 11, FP 2, FN 7, còn vòng 1 có TP 6, FP 0, FN 12. Trên `frame_0150`, kết quả đổi từ TP 10, FP 2, FN 10 thành TP 4, FP 0, FN 16. Như vậy mô hình sau train thận trọng hơn nhưng bỏ sót thêm nhiều xe, đặc biệt các box xa và khó thấy; ảnh so sánh không cung cấp bằng chứng về một cải thiện recall.

Cần tách ba loại bằng chứng. `BLIND_SCAN.md` là quan sát độc lập trước khi xem pre-label: ở `frame_0107.jpg` quan sát khoảng 26 xe và ghi nhận xe xa chỉ thấy đèn, xe tối hoặc bị che là ca khó. `REVIEW_LOG.csv` và `round1_diff.md` phản ánh lỗi pre-label đã được người dùng sửa, gồm 174 box thêm và 14 box xóa. Còn `metrics_round1.json` cùng `compare_round1.jpg` là kết quả mô hình sau train trên tập test. Một ca khó theo guideline là vùng chỉ có ánh sáng/phản chiếu hoặc vài điểm đèn: cần đủ căn cứ nhận ra phương tiện bốn bánh mới gán `car`; nếu vẫn nhận ra thân xe nhưng bị che một phần thì box nên ôm phần xe nhìn thấy thay vì suy diễn toàn bộ vật thể.

## 5. Kết luận và giới hạn

Vòng 1 không tốt hơn cold start trên bộ tham chiếu hiện tại: AP50 giảm 0.2819, recall và F1 giảm mạnh, dù precision tại ngưỡng 0.25 tăng và FP về 0. Chỉ fine-tune bằng 12 ảnh và 329 box là phạm vi rất nhỏ; hơn nữa, các ảnh đến từ video đường cao tốc ban đêm nên nhiều frame gần nhau có nguy cơ near-duplicate, làm độ đa dạng hiệu dụng còn thấp hơn con số 12. Trước khi train thêm, cần kiểm tra lại chất lượng/chính sách nhãn train và test, phân bố kích thước box, cấu hình fine-tune, confidence và hiện tượng mô hình trở nên quá dè dặt sau train.

Giới hạn quan trọng nhất là tập test chỉ có 20 ảnh và nhãn tham chiếu do mô hình tạo, chưa được con người rà từng box. Vì vậy, AP50 ở đây chỉ phản ánh mức khớp với bộ tham chiếu này, không chứng minh chất lượng thực địa hay coi nhãn test là chân lý tuyệt đối. Luật bỏ qua 14 box cao dưới 16 px cũng khiến kết quả không mô tả đầy đủ khả năng phát hiện mọi xe cực nhỏ. Với cỡ mẫu nhỏ, vài frame hoặc vài nhãn tham chiếu chưa chính xác có thể làm chỉ số thay đổi đáng kể.

Nếu tiếp tục vòng 2, nên ưu tiên hai nhóm ca: (1) xe xa/nhỏ và thiếu sáng, do recall small đã về 0.0000 và review cho thấy nhiều box bị bỏ sót; (2) xe bị che một phần hoặc chỉ hiện qua tín hiệu thị giác yếu, vì cần người phân biệt xe thật với glare/reflection. Hai nhóm này có chi phí rà nhãn cao do phải phóng to, xác minh từng đối tượng và đặt box cẩn thận. Việc chọn mẫu nên kết hợp uncertainty với diversity/de-duplication và khoảng cách thời gian, tránh lấy dày các frame gần nhau. Đây là hướng thử nghiệm có căn cứ từ pattern lỗi, nhưng không bảo đảm vòng 2 chắc chắn sẽ tăng AP50.
