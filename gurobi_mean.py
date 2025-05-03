import csv

filename = '/export2/curiekim/pCQO-mis-benchmark/zero_to_stage_128_of_128_total_stages_2025-05-01 10:01:41.852414_ER700800_Gurobi30sec.csv'
column_index = 2 

values = []
with open(filename, 'r') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        values.append(float(row[column_index]))

average = sum(values) / len(values)
print(f"Average of column {column_index}: {average}")