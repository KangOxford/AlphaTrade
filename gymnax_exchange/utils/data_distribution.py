

import os

folder_path = "/home/myuser/data/rawLOBSTER/AMZN/2024"
output_path = "/home/myuser/gymnax_exchange/jaxen/Results/message_file_row_counts.txt"


total_rows = 0
output_lines = []

for filename in os.listdir(folder_path):
    if "message" in filename and filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            row_count = sum(1 for _ in f)
            print(f"{filename}: {row_count} rows")
            output_lines.append(f"{filename}: {row_count} rows")
            total_rows += row_count

output_lines.append(f"Total rows in all message files: {total_rows}")
print(f"Total rows in all message files: {total_rows}")

with open(output_path, "w") as out_file:
    for line in output_lines:
        out_file.write(line + "\n")