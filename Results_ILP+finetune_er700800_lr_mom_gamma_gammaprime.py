import os
import sys

if len(sys.argv) != 6:
    print("Usage: python script.py <graph_name> <lr> <mom> <gamma> <gamma_prime>")
    print("e.g: python script.py er_700-800 0.00001 0.7 500 7")
    sys.exit(1)

subdir = sys.argv[1]
lr = sys.argv[2]
mom = sys.argv[3]
gamma = sys.argv[4]
gamma_prime = sys.argv[5]

log_dir = f"/export3/curiekim/fobc/ILP+fast_tune/{subdir}_{lr}_{mom}_{gamma}_{gamma_prime}/"
prefix = "ER_700_800_0.15_"
suffix = ".log"
suffix_gurobi = "_GurobiInit.log"
values = []
times = []
values_gurobi = []
times_gurobi = []

for num in range(128):
    filename = f"{prefix}{num}{suffix}"
    filename_gurobi = f"{prefix}{num}{suffix_gurobi}"
    filepath = os.path.join(log_dir, filename)
    filepath_gurobi = os.path.join(log_dir, filename_gurobi)

    if not os.path.isfile(filepath):
        continue  #skip if file is missing

    if not os.path.isfile(filepath_gurobi):
        continue  #skip if file is missing

    with open(filepath, "r") as f:
        lines = f.readlines()
    for i in range(1, len(lines)):
        # val = int(lines[-4].strip())
        # values.append(val)
        if lines[i].strip().startswith("30."):
            try:
                val = int(lines[i - 1].strip())
                values.append(val)
                time = float(lines[i].strip())
                times.append(time)
            except ValueError:
                print(f"Cannot parse int in line before 30. in file {filename}")
            break
        
    with open(filepath_gurobi, "r") as f:
        lines = f.readlines()

    for i in range(1, len(lines)):
        if lines[i].strip().startswith("30."):
            try:
                val_gurobi = int(lines[i - 1].strip())
                values_gurobi.append(val_gurobi)
                time_gurobi = float(lines[i].strip())
                times_gurobi.append(time_gurobi)
            except ValueError:
                print(f"Cannot parse int in line before 30. in file {filepath_gurobi}")
            break 



if values:
    average = sum(values) / len(values)
    average_time = sum(times) / len(times)
    print(f"Average pCQO MIS/time with [{lr}, {mom}, {gamma}, {gamma_prime}] among {len(values)} datasets:", average, average_time)
else:
    print("No matching values found.")

if values_gurobi:
    average_gurobi = sum(values_gurobi) / len(values_gurobi)
    average_time_gurobi = sum(times_gurobi) / len(times_gurobi)
    print(f"Average GurobiInit+pCQO MIS/time with [{lr}, {mom}, {gamma}, {gamma_prime}] among {len(values_gurobi)} datasets:", average_gurobi, average_time_gurobi)
else:
    print("No matching values found.")