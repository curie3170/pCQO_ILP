import csv
 
filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_384_of_384_total_stages_2025-05-03 03:45:02.098629_ER700800_fasttune(unfixed)133200steps_Gurobi.csv'
filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_384_of_384_total_stages_2025-05-05 05:58:14.422408_ER700800_fasttune(repeated_fixed)133200steps_Gurobi.csv'
pcqo_450step_values = []
pcqo_30s_values = []
pcqo_450step_gurobi_values = []
pcqo_30s_gurobi_values = []

pcqo_450step_time = []
pcqo_30s_time = []
pcqo_450step_gurobi_time = []
pcqo_30s_gurobi_time = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader) 

    for row in reader:
        dataset_name = row[1]
        if "_0.15_" in dataset_name:
            try:
                col3 = float(row[2])
                col31 = float(row[31])
                col32 = float(row[32])
                col33 = float(row[33])
                pcqo_450step_values.append(col3)
                pcqo_30s_values.append(col31)
                pcqo_450step_gurobi_values.append(col32)
                pcqo_30s_gurobi_values.append(col33)
                
                col34 = float(row[34])
                col63 = float(row[63])
                col64 = float(row[64])
                col65 = float(row[65])
                pcqo_450step_time.append(col34)
                pcqo_30s_time.append(col63)
                pcqo_450step_gurobi_time.append(col64)
                pcqo_30s_gurobi_time.append(col65)
            except ValueError:
                continue  

if pcqo_450step_values and pcqo_30s_values:
    avg_pcqo_450step_mis = sum(pcqo_450step_values) / len(pcqo_450step_values)
    avg_pcqo_30s_mis = sum(pcqo_30s_values) / len(pcqo_30s_values)
    avg_pcqo_450step_gurobi_mis = sum(pcqo_450step_gurobi_values) / len(pcqo_450step_gurobi_values)
    avg_pcqo_30s_gurobi_mis = sum(pcqo_30s_gurobi_values) / len(pcqo_30s_gurobi_values)

    avg_pcqo_450step_time = sum(pcqo_450step_time) / len(pcqo_450step_time)
    avg_pcqo_30s_time = sum(pcqo_30s_time) / len(pcqo_30s_time)
    avg_pcqo_450step_gurobi_time = sum(pcqo_450step_gurobi_time) / len(pcqo_450step_gurobi_time)
    avg_pcqo_30s_gurobi_time = sum(pcqo_30s_gurobi_time) / len(pcqo_30s_gurobi_time)

    print(f"pcqo_450step MIS/ time: {avg_pcqo_450step_mis}, {avg_pcqo_450step_time}")
    print(f"pcqo_30s MIS/ time: {avg_pcqo_30s_mis}, {avg_pcqo_30s_time}")
    print(f"pcqo_450step+Gurobi MIS/ time: {avg_pcqo_450step_gurobi_mis}, {avg_pcqo_450step_gurobi_time}")
    print(f"pcqo_30s+Gurobi MIS/ time: {avg_pcqo_30s_gurobi_mis}, {avg_pcqo_30s_gurobi_time}")
else:
    print("No matching rows found.")
    
