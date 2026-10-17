import csv
import subprocess

out = subprocess.check_output(["tasklist", "/v", "/fo", "csv"], text=True)
reader = csv.reader(out.strip().splitlines())
header = next(reader)

for row in reader:
    if "python" in row[0].lower():
        print(f"PID: {row[1]} | Image: {row[0]} | Mem: {row[4]} | User: {row[6]} | CPU: {row[7]} | Window: {row[8]}")
