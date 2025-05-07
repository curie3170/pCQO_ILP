import csv
 
filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_70_of_70_total_stages_2025-05-01 10:40:36.349801_ER1000_Gurobi30sec.csv'
filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_70_of_70_total_stages_2025-05-01 11:25:37.777037_ER1500_Gurobi30sec.csv'
filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_70_of_70_total_stages_2025-05-01 12:32:04.934927_ER2500_Gurobi30sec.csv'
filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_70_of_70_total_stages_2025-05-01 13:58:04.935475_ER3000_Gurobi30sec.csv'
filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_70_of_70_total_stages_2025-05-06 12:58:02.853824_ER2000_Gurobi30sec.csv'

col3_values = []
col4_values = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader) 

    for row in reader:
        dataset_name = row[1]
        if "_0.10_" in dataset_name:
            try:
                col3 = float(row[2])
                col4 = float(row[3])
                col3_values.append(col3)
                col4_values.append(col4)
            except ValueError:
                continue  

if col3_values and col4_values:
    avg_col3 = sum(col3_values) / len(col3_values)
    avg_col4 = sum(col4_values) / len(col4_values)

    print(f"Solution Size: {avg_col3}")
    print(f"Solution Time: {avg_col4}")
else:
    print("No matching rows found.")
    

col3_values = []
col4_values = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 스킵

    for row in reader:
        dataset_name = row[1]
        if "_0.20_" in dataset_name:
            try:
                col3 = float(row[2])
                col4 = float(row[3])
                col3_values.append(col3)
                col4_values.append(col4)
            except ValueError:
                continue  

if col3_values and col4_values:
    avg_col3 = sum(col3_values) / len(col3_values)
    avg_col4 = sum(col4_values) / len(col4_values)

    print(f"Solution Size: {avg_col3}")
    print(f"Solution Time: {avg_col4}")
else:
    print("No matching rows found.")
    
    
col3_values = []
col4_values = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 스킵

    for row in reader:
        dataset_name = row[1]
        if "_0.30_" in dataset_name:
            try:
                col3 = float(row[2])
                col4 = float(row[3])
                col3_values.append(col3)
                col4_values.append(col4)
            except ValueError:
                continue  

if col3_values and col4_values:
    avg_col3 = sum(col3_values) / len(col3_values)
    avg_col4 = sum(col4_values) / len(col4_values)

    print(f"Solution Size: {avg_col3}")
    print(f"Solution Time: {avg_col4}")
else:
    print("No matching rows found.")
    
    
col3_values = []
col4_values = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 스킵

    for row in reader:
        dataset_name = row[1]
        if "_0.40_" in dataset_name:
            try:
                col3 = float(row[2])
                col4 = float(row[3])
                col3_values.append(col3)
                col4_values.append(col4)
            except ValueError:
                continue  

if col3_values and col4_values:
    avg_col3 = sum(col3_values) / len(col3_values)
    avg_col4 = sum(col4_values) / len(col4_values)

    print(f"Solution Size: {avg_col3}")
    print(f"Solution Time: {avg_col4}")
else:
    print("No matching rows found.")
    
    
col3_values = []
col4_values = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 스킵

    for row in reader:
        dataset_name = row[1]
        if "_0.50_" in dataset_name:
            try:
                col3 = float(row[2])
                col4 = float(row[3])
                col3_values.append(col3)
                col4_values.append(col4)
            except ValueError:
                continue  

if col3_values and col4_values:
    avg_col3 = sum(col3_values) / len(col3_values)
    avg_col4 = sum(col4_values) / len(col4_values)

    print(f"Solution Size: {avg_col3}")
    print(f"Solution Time: {avg_col4}")
else:
    print("No matching rows found.")
    
    
col3_values = []
col4_values = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 스킵

    for row in reader:
        dataset_name = row[1]
        if "_0.60_" in dataset_name:
            try:
                col3 = float(row[2])
                col4 = float(row[3])
                col3_values.append(col3)
                col4_values.append(col4)
            except ValueError:
                continue  

if col3_values and col4_values:
    avg_col3 = sum(col3_values) / len(col3_values)
    avg_col4 = sum(col4_values) / len(col4_values)

    print(f"Solution Size: {avg_col3}")
    print(f"Solution Time: {avg_col4}")
else:
    print("No matching rows found.")
    
    
col3_values = []
col4_values = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 스킵

    for row in reader:
        dataset_name = row[1]
        if "_0.70_" in dataset_name:
            try:
                col3 = float(row[2])
                col4 = float(row[3])
                col3_values.append(col3)
                col4_values.append(col4)
            except ValueError:
                continue  

if col3_values and col4_values:
    avg_col3 = sum(col3_values) / len(col3_values)
    avg_col4 = sum(col4_values) / len(col4_values)

    print(f"Solution Size: {avg_col3}")
    print(f"Solution Time: {avg_col4}")
else:
    print("No matching rows found.")