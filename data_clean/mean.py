import csv

filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_128_of_128_total_stages_2025-05-01 10:01:41.852414_ER700800_Gurobi30sec.csv'  
filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_70_of_70_total_stages_2025-05-01 10:40:36.349801_ER1000_Gurobi30sec.csv'

mis_values = []
time_values = []

with open(filename, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        mis_values.append(float(row[2]))  # 3번째 컬럼
        time_values.append(float(row[3]))  # 4번째 컬럼

mis = sum(mis_values) / len(mis_values)
time = sum(time_values) / len(time_values)

print(f"Solution Size: {mis}")
print(f"Solution Time: {time}")