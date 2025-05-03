import os

log_dir = "/export2/curiekim/fobc/fast_tune+ILP/er_700-800"
prefix = "ER_700_800_0.15_"
suffix = "_tuningiter5000.log"

values = []

for num in range(128):
    filename = f"{prefix}{num}{suffix}"
    filepath = os.path.join(log_dir, filename)

    if not os.path.isfile(filepath):
        continue  #skip if file is missing

    with open(filepath, "r") as f:
        lines = f.readlines()

    for i in range(1, len(lines)):
        if "30." in lines[i]:
            try:
                val = int(lines[i - 1].strip())
                values.append(val)
            except ValueError:
                print(f"Cannot parse int in line before 30. in file {filename}")
            break 
    # val = int(lines[-3].strip())
    # values.append(val)

if values:
    average = sum(values) / len(values)
    print("Num of data:", len(values))
    print("Extracted values:", values)
    print("Average:", average)
else:
    print("No matching values found.")