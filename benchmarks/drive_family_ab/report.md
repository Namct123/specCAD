# Benchmark họ Modular Flanged Bearing Drive Unit

A dùng ba phiên riêng; L và S nhận nguyên mã A–M. B dùng một phiên cho M, rồi đổi tham số L/S cục bộ.
Phiên độc lập không tắt được cache tiền tố phía nhà cung cấp; báo cáo ghi cache thực tế và cả tổng tính cache.
Trục hai lỗ ngang của mỗi chân đế lấy tại giữa chiều cao. Kiểm tra Boolean độc lập trên tất cả cặp của STEP xuất lại.

| Nhánh | Size | Phiên | Lượt Cloud | Chưa cache + output | Tính cả cache | Dựng + kiểm định cục bộ | Solid | BRep |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | M | 1 | 1 | 13472 | 28704 | 7.00 s | 40 | 780 cặp / 0.0000 mm³ |
| A | L | 1 | 1 | 13846 | 29078 | 9.05 s | 56 | 1540 cặp / 0.0000 mm³ |
| A | S | 1 | 1 | 13848 | 29080 | 6.91 s | 32 | 496 cặp / 0.0000 mm³ |
| B | M | 1 | 1 | 14175 | 29407 | 30.56 s | 40 | 780 cặp / 0.0000 mm³ |
| B | L | 0 | 0 | 0 | 0 | 18.29 s | 56 | 1540 cặp / 0.0000 mm³ |
| B | S | 0 | 0 | 0 | 0 | 16.25 s | 32 | 496 cặp / 0.0000 mm³ |

| Tổng | Phiên | Lượt Cloud | Chưa cache + output | Tính cả cache | Cục bộ | Đạt | MCP |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 3 | 41166 | 86862 | 22.96 s | 3/3 | 0 |
| B | 1 | 1 | 14175 | 29407 | 65.11 s | 3/3 | 7 |

Hybrid giảm **65.57%** token chưa cache + output, và **66.15%** token khi tính cả cache.

B–M biên dịch một lần và gọi lắp ghép hai lần. Lần đầu bị từ chối vì vị trí lắp là danh sách. Lần sau đổi danh sách sang ánh xạ có tên, dùng lại các STEP đã biên dịch và giữ nguyên toàn bộ thông số. Thời gian và số lượt MCP đã tính cả lần lỗi; không có thêm lượt Cloud.

[Số đo đã lược bỏ đường dẫn và log phiên](result.json)
