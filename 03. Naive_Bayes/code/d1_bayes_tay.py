"""Bước 1: tính xác suất hậu nghiệm bằng công thức Bayes."""


def bayes(prior, likelihood, reverse_likelihood):
    """Trả về P(A|B) từ P(A), P(B|A) và P(B|không A)."""
    numerator = likelihood * prior
    denominator = numerator + reverse_likelihood * (1 - prior)
    return numerator / denominator


print("=== Ví dụ xét nghiệm bệnh ===")
p_benh = 0.01
p_duong_tinh_khi_benh = 0.99
p_duong_tinh_khi_khong_benh = 0.01
print(f"P(Bệnh | Dương tính) = {bayes(p_benh, p_duong_tinh_khi_benh, p_duong_tinh_khi_khong_benh):.4f}")

print("\n=== Ảnh hưởng của xác suất tiên nghiệm ===")
for prior in [0.001, 0.01, 0.05, 0.10, 0.50]:
    posterior = bayes(prior, 0.99, 0.01)
    print(f"P(Bệnh) = {prior:>5.3f} -> P(Bệnh | Dương tính) = {posterior:.4f}")

print("\n=== Ví dụ dây chuyền sản xuất ===")
# A làm 80% sản phẩm và có 1% lỗi; B làm 20% sản phẩm và có 5% lỗi.
p_b_khi_loi = bayes(prior=0.20, likelihood=0.05, reverse_likelihood=0.01)
print(f"P(Dây chuyền B | Sản phẩm lỗi) = {p_b_khi_loi:.4f}")

loi_a = 800 * 0.01
loi_b = 200 * 0.05
print(f"Kiểm tra trên 1000 sản phẩm: B có {loi_b:.0f}/{loi_a + loi_b:.0f} sản phẩm lỗi")
