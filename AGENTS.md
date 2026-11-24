# Hybrid CAD

## Một manifest cho mỗi cụm

- Dùng `run_hybrid_batch(parts_specs, assembly_plan=...)` một lần khi đã đủ thông tin. `parts_specs` chứa mỗi hình học duy nhất một lần. `assembly_plan` chứa `export_name`, các vị trí và `expected_volume` khi có giá trị tính độc lập.
- Định nghĩa lặp dùng `part` trong từng placement hoặc `circular_patterns` cho vòng chia XY. Mỗi pattern được đánh số theo tên, với `part`, `count`, `pcd`, `center` XYZ mặc định [0, 0, 0] và `start_angle` theo độ dương ngược chiều kim đồng hồ.
- Khi một cụm có hình học ngoài khả năng compiler, đặt `cloud_parts_script` trong cùng manifest. Script tạo dict `cloud_parts` cho những tên trong `pending_parts`. Nếu thiếu script và batch báo ESCALATE, giải quyết toàn bộ chi tiết còn thiếu trong một lượt.

## Phân luồng

- `compiler_cpu` dựng chi tiết tham số theo số đã biết. Dùng `cone` với `bottom_radius`, `top_radius`, `height` cho nón tròn và `tube` với `od`, `id`, `h` cho ống đồng tâm. Ghi thẳng số vào `base`.
- `local_sft` với `raw_text` và `geometry_kind: "cone"` dành cho mô tả nguyên văn từ người dùng hoặc BOM. Dòng nón có nhãn đường kính hoặc bán kính đáy, đỉnh và chiều cao theo mm được compiler xử lý trực tiếp. Các cách viết khác đi qua Ollama. Gọi cùng một batch cho nhiều dòng; chế độ chạy MCP cục bộ không cần Codex. Dùng `max_sft_retries: 0` khi đo hiệu năng. Tuyến sinh mã SFT cũ dành cho hình học đơn nguyên công đủ điều kiện và có hợp đồng hình học.
- `cloud_direct` dành cho BSpline phức hợp, loft thành mỏng, sweep 3D và các hình học ngoài khả năng compiler. Gọi tool MCP trực tiếp và dùng trường lỗi gọn trả về để xử lý. Xem log cục bộ khi cần chi tiết.

## Nghiệm thu

- Gốc mặc định `center_bottom`. Giữ đáy tại Z=0 trong dung sai 0,05 mm. So thể tích với `max(1.0 mm³, 0.001 × V_expected)`. Khai báo thể tích, hộp bao và vùng rỗng độc lập khi có.
- Chỉ báo nghiệm thu toàn cụm khi xác minh va chạm hoàn tất, thể tích đạt và các artifact xuất thành công. Cặp vượt giới hạn phức tạp BRep là chưa xác minh.
- Đo A/B trên cùng đề và tiêu chuẩn nghiệm thu. Lấy Cloud token từ bộ đếm phiên là `input_tokens - cached_input_tokens + output_tokens`. Ghi riêng số byte payload, thời gian Cloud, thời gian dựng và kiểm định.
