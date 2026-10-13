"""Bài 1: áp dụng định lý Bayes cho ba nhóm sinh viên."""

priors = {"Nhóm 1": 0.50, "Nhóm 2": 0.30, "Nhóm 3": 0.20}
pass_rates = {"Nhóm 1": 0.95, "Nhóm 2": 0.70, "Nhóm 3": 0.30}

evidence = sum(priors[group] * pass_rates[group] for group in priors)
posteriors = {
    group: priors[group] * pass_rates[group] / evidence for group in priors
}

print(f"P(qua môn) = {evidence:.4f}")
for group, probability in posteriors.items():
    print(f"P({group} | qua môn) = {probability:.4f}")
print(f"Tổng ba xác suất hậu nghiệm = {sum(posteriors.values()):.4f}")
print("Giải thích: nhóm 1 chiếm tỷ lệ ban đầu lớn và có xác suất qua môn cao nhất,")
print("nên một sinh viên đã qua môn có khả năng thuộc nhóm 1 cao nhất.")
