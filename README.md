# Dự đoán World Cup 2030

Dự án hướng đến xây dựng mô hình dự đoán kết quả và nhà vô địch World Cup 2030. Dữ liệu và 104 trận World Cup 2026 được dùng làm giai đoạn thử nghiệm để xây dựng đặc trưng Elo, huấn luyện và đánh giá mô hình dự đoán tỉ số. Mô hình mô phỏng toàn giải để ước tính xác suất vô địch World Cup 2030 chưa được triển khai.

## Trạng thái hiện tại

Dự án có hai notebook, phải chạy theo đúng thứ tự:

1. [`notebooks/wc.ipynb`](./notebooks/wc.ipynb) — xử lý dữ liệu và tạo đặc trưng.
2. [`notebooks/train.ipynb`](./notebooks/train.ipynb) — huấn luyện và so sánh bốn mô hình.

### `wc.ipynb`

1. Đọc dữ liệu lịch sử trận đấu quốc tế và dữ liệu 104 trận World Cup 2026.
2. Chuẩn hoá tên đội giữa hai nguồn dữ liệu (`Czechia` → `Czech Republic`, `Türkiye` → `Turkey`, `IR Iran` → `Iran`, `USA` → `United States`, …).
3. Lọc lịch sử từ 14/06/2018 đến trước 11/06/2026, loại các giải CONIFA.
4. Tính Elo tuần tự trên toàn bộ lịch sử từ 1930, lấy giá trị **trước** mỗi trận. Elo của 104 trận World Cup 2026 cũng được cập nhật lần lượt qua từng trận, giống hệt cách tính lúc huấn luyện.
5. Tạo 4 đặc trưng phong độ: bàn ghi và bàn thủng trung bình trong tối đa 5 trận trước đó. Các trận cùng ngày không dùng kết quả của nhau.
6. Xuất dữ liệu và đặc trưng vào thư mục `loc/`.

Bảy feature đầu vào mô hình: `home_elo_pre`, `away_elo_pre`, `neutral`, và 4 cột `*_form_goals_*_last5`.

### `train.ipynb`

Huấn luyện bốn mô hình (Poisson Regression, Ridge, Random Forest, Extra Trees) để dự đoán số bàn của hai đội, rồi dựng ma trận xác suất tỉ số từ hai giá trị λ.

Chỉ số đánh giá, theo thứ tự ưu tiên:

- **RPS** — chất lượng xác suất thắng/hoà/thua, càng thấp càng tốt.
- **Đúng K/H/T** — tỉ lệ đoán đúng kết quả trận.
- **Đúng tỉ số** — chỉ số tham khảo; trần thực tế chỉ khoảng 15%.

Notebook có thêm hai baseline (luôn đoán 1-1, luôn đoán đội nhà thắng) và cột "số tỉ số khác nhau" để phát hiện trường hợp mô hình chỉ dự đoán vài tỉ số cố định.

Notebook kiểm định lại bằng walk-forward qua 12 giải: mỗi giải chỉ huấn luyện trên các trận trước ngày khai mạc. Kết quả WC 2026: Poisson RPS 0.1602 (66.3% đúng K/H/T), Ridge 0.1587 (66.3%), MNLogit 0.1566 (66.3%). Trên 614 trận walk-forward: MNLogit 0.1824, Poisson 0.1834, Ridge 0.1852.

## Cấu trúc dự án

```text
WorldCup2030/
├── dataset/
│   ├── du_lieu_tho/
│   │   ├── former_names.csv
│   │   ├── match_prediction_features.csv
│   │   └── results.csv
│   └── mo_phong/
│       ├── FIFA_confederations.csv
│       ├── fixtures_knockout_wc2026.csv
│       └── group_stages.csv
├── loc/
│   ├── matches_truoc_va_sau_wc_2026.csv
│   ├── matches_truoc_wc_2026_da_loc.csv
│   ├── matches_truoc_wc_2026_elo_features.csv
│   └── wc_2026_da_loc.csv
└── notebooks/
    ├── train.ipynb
    └── wc.ipynb
```

`dataset/du_lieu_tho/` chứa dữ liệu đầu vào về kết quả trận đấu và thông tin trận World Cup 2026. `dataset/mo_phong/` chứa dữ liệu bảng đấu, lịch vòng loại trực tiếp và liên đoàn. Các file trong `loc/` là dữ liệu trung gian hoặc đầu ra do notebook tạo ra.

## Cài đặt và chạy

Cần Python 3, Jupyter Notebook, `pandas`, `numpy`, `scipy` và `scikit-learn`:

```powershell
python -m pip install pandas numpy scipy scikit-learn notebook
```

Mở notebook với thư mục làm việc là `notebooks`, vì các đường dẫn dữ liệu trong notebook được viết tương đối từ thư mục đó:

```powershell
Set-Location .\notebooks
jupyter notebook wc.ipynb
```

Chạy `wc.ipynb` hết các ô trước, sau đó mở và chạy `train.ipynb`. Trong VS Code, cũng có thể mở từng notebook, chọn kernel Python đã cài đủ gói ở trên, rồi chạy các ô theo thứ tự từ trên xuống dưới.

## Các file đầu ra chính

- `loc/matches_truoc_wc_2026_da_loc.csv`: lịch sử trận đấu sau khi lọc, kèm 4 đặc trưng phong độ.
- `loc/wc_2026_da_loc.csv`: 104 trận World Cup 2026 với đặc trưng phong độ và Elo trước từng trận.
- `loc/matches_truoc_va_sau_wc_2026.csv`: lịch sử kết hợp với dữ liệu World Cup 2026, đầy đủ đặc trưng.
- `loc/matches_truoc_wc_2026_elo_features.csv`: lịch sử trận đấu kèm đặc trưng Elo trước trận.

## Hướng phát triển

- Thêm feature mới để vượt trần hiện tại: xG (expected goals), số ngày nghỉ, hoặc tỉ lệ cược của nhà cái. Đổi mô hình gần như không cải thiện thêm.
- Đã thử thêm giá trị đội hình theo Transfermarkt: chỉ cải thiện ΔRPS khoảng 0.003, chưa bằng một phần tư sai số chuẩn của tập test, nên đã bỏ.
- Dùng dự đoán số bàn để mô phỏng vòng bảng và vòng loại trực tiếp World Cup 2030.
- Ước tính xác suất vô địch cho từng đội bằng mô phỏng Monte Carlo.
