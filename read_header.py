import csv

with open("source_matcha_pro_all/combined_data.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

print(header)


# import csv
# from itertools import islice

# with open("result/match_sbr_kdm.csv", newline="", encoding="utf-8") as f:
#     reader = csv.reader(f)
#     header = next(reader)  # skip header
#     first_10_rows = list(islice(reader, 10))

# print(first_10_rows)

