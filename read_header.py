import csv

with open("result/match_sbr_kdm.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

print(header)
