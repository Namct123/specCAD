# Chạy Hybrid CAD với Codex và Ollama

Cấu hình `local-cad` để chạy Python trong môi trường ảo với `-m mcp_server.server` và thư mục làm việc là thư mục gốc repo. Dùng `run_hybrid_batch` để dựng chi tiết, lắp ghép và xuất cụm trong cùng một yêu cầu.

## Manifest cho cụm có chi tiết lặp

```json
{
  "parts_specs": [
    {"name": "plate", "tier": "compiler_cpu", "base": {"shape": "tube", "od": 260, "id": 100, "h": 8}, "features": [{"op": "hole_pcd", "d": 6.6, "pcd": 210, "count": 40, "start_angle": 11.25, "sel": ">Z"}]},
    {"name": "pin", "tier": "compiler_cpu", "base": {"shape": "cylinder", "d": 6, "h": 16}}
  ],
  "assembly_plan": {
    "export_name": "flange40",
    "placements": {"plate": {"translation": [0, 0, 0]}},
    "circular_patterns": {"pin": {"part": "pin", "count": 40, "pcd": 210, "center": [0, 0, -4], "start_angle": 11.25}}
  }
}
```

Gửi số đã biết ngay trong `base`. Nón tròn dùng `{"shape":"cone","bottom_radius":18,"top_radius":3,"height":40}`. Bổ sung `expected_volume` nếu đã tính độc lập. Dùng `cloud_parts_script` trong manifest cho hình học ngoài khả năng compiler và kiểm tra `collision_verification_complete` trong kết quả.

## Mô tả nguyên văn và BOM

Gửi các dòng nón tròn chưa phân tích trong cùng `parts_specs`, ví dụ `{"name":"cone_a","tier":"local_sft","geometry_kind":"cone","raw_text":"Nón cụt đáy Ø36 mm, đỉnh Ø6 mm, cao 40 mm."}`. Chạy bằng MCP client cục bộ để dùng 0 token Cloud. Dòng đủ nhãn như “Nón cụt, đường kính đáy 24 mm, đường kính đỉnh 12 mm, cao 18 mm.” được quy đổi trực tiếp sang bán kính và dựng trên CPU. Các cách viết ngoài mẫu dùng Ollama qua `127.0.0.1:11434`, mặc định `qwen2.5:1.5b`. `CAD_OLLAMA_RECIPE_MODEL` đổi model mặc định. Trường `model` yêu cầu dùng model chỉ định cho dòng đó. Dùng `max_sft_retries: 0` để giới hạn một lần thử. Nếu có số đo độc lập, truyền `expected_volume`, `expected_z_range` hoặc `expected_end_radii_mm` để kiểm tra trích xuất. Với đầu vào chỉ có lời mô tả, cần đối chiếu kích thước trích xuất trước khi dùng để chế tạo.

## Đo A/B

Đọc [báo cáo benchmark họ drive unit](benchmarks/drive_family_ab/report.md). Ba manifest tái tạo nhánh Hybrid nằm trong [examples/drive_family](examples/drive_family/).

## Điều phối API trực tiếp

Chạy `python hybrid_orchestrator.py de_bai.txt --mode cloud --model TEN_MODEL --output workspace/run1` với `ANTHROPIC_API_KEY` đã thiết lập trong môi trường. Bộ điều phối gửi schema gọn, đề bài và tối đa một yêu cầu sửa nếu bật `--cloud-repair`. Token lấy từ usage của Messages API. Dùng `--mode local` để lập manifest bằng Ollama, hoặc `--mode manifest` để chạy tệp JSON sẵn có mà không gọi Cloud.

`assembly_plan.linear_series` ánh xạ tên nhóm sang `n`, `xyz`, `step` và `part`. Với dãy có kích thước biến thiên, dùng `spec` thay cho `part` và `vary` để chỉ định lượng tăng cho từng trường số, ví dụ `{"base.sections.1.d":1}`. Mỗi dãy mở rộng thành vị trí và định nghĩa bình thường. Compiler hỗ trợ `loft_cone_hollow`, `sweep_pipe_arc` và `mounting_foot_cutout`.

`--repair-policy` đọc một tệp do người vận hành xác nhận, ánh xạ tên vị trí rõ ràng sang danh sách cao độ Z được phép. Khi xác minh có va chạm, một lựa chọn được thử tất định. Ollama chọn giữa nhiều phương án và hệ thống chạy lại toàn bộ kiểm định. `--cloud-repair` cho phép một lượt Cloud chọn phương án còn lại hoặc vá đoạn mã đã được xác định tên. Mã CadQuery tùy ý chỉ chạy khi chủ động bật `--allow-generated-code`. Kiểm tra manifest trước khi bật cho mã mới. Tiến trình CadQuery riêng cách ly lỗi tiến trình và vẫn có quyền của tài khoản hiện tại.

File `report.md` sinh từ kết quả cục bộ cùng `result.json`. Dung lượng manifest tính theo byte, tách riêng khỏi token Cloud.

## CLI Hybrid CAD v4

- Chạy manifest YAML hoặc JSON qua `python orchestrator/hybrid_orchestrator.py manifest.yaml --output workspace/run_v4`. CLI cũ `hybrid_orchestrator.py` vẫn dùng được.
- Lập manifest qua Cloud bằng `--mode cloud --prompt "..." --output workspace/run_cloud`. Chọn `--provider openai`, `openrouter` hoặc `anthropic`. Đặt `HYBRID_CLOUD_MODEL` và khóa tương ứng `OPENAI_API_KEY`, `OPENROUTER_API_KEY` hoặc `ANTHROPIC_API_KEY`. Có thể đổi điểm cuối tương thích OpenAI bằng `OPENAI_BASE_URL`. Đầu ra có `assembly_plan_yaml` và `cloud_direct_parts` trong cùng một gói. Chỉ bật `--allow-generated-code` với mã đã được review hoặc từ nguồn được tin cậy.
- Chạy `--local-only --prompt "..." --output workspace/run_local` để Ollama lập manifest tham số. `CAD_OLLAMA_RECIPE_MODEL` đặt model mặc định và `--local-model` ghi đè theo lượt. Đặt `--export-name` để khóa tên xuất tệp theo yêu cầu.
- `--repair-policy` nhận YAML hoặc JSON do người dùng cung cấp với tên instance và danh sách vị trí XYZ tuyệt đối được phép. Giá trị số cũ vẫn có nghĩa là cao độ Z. `ENABLE_LOCAL_REPAIR=0` tắt đường tự sửa cục bộ. `--cloud-repair` bật tối đa một lần vá phần hỏng. V4 luôn chạy Boolean BRep cho các cặp qua lọc AABB, kể cả cặp có nhiều mặt. Nếu phép Boolean không xác minh được hoặc worker hết thời gian, lượt chạy chưa đạt.
- `hollow_frustum_loft` hỗ trợ `axis: +X / +Y / +Z`. `swept_tube_path` nhận danh sách `points` XYZ và `bend_radius`, với `id: 0` cho ống đặc. `bracket_with_cutout` nhận `bolt_holes` và `cuts` trụ, hộp hoặc mặt phẳng. `expected_global_bbox` kiểm tra thứ tự `[xmin,ymin,zmin,xmax,ymax,zmax]` sau xoay và tịnh tiến.
- `result.json` ghi số lượt API, token input/cache/output/tổng, thời gian từng tầng và trạng thái kiểm định.
