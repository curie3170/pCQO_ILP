import csv

filename = '/export3/curiekim/pCQO-mis-benchmark/meaningful_results/zero_to_stage_512_of_512_total_stages_2025-05-03 14:05:16.664202_ER700800_KaMIS+ILP.csv'
Kamis_0sec_values = []
Kamis_1sec_values = []
Kamis_0sec_gurobi_values = []
Kamis_1sec_gurobi_values = []

Kamis_0sec_time = []
Kamis_1sec_time = []
Kamis_0sec_gurobi_time = []
Kamis_1sec_gurobi_time = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    next(reader) 

    for row in reader:
        dataset_name = row[1]
        if "_0.15_" in dataset_name:
            try:
                col2 = float(row[2])
                col3 = float(row[3])
                col4 = float(row[4])
                col5 = float(row[5])
                Kamis_0sec_values.append(col2)
                Kamis_1sec_values.append(col3)
                Kamis_0sec_gurobi_values.append(col4)
                Kamis_1sec_gurobi_values.append(col5)
                
                col6 = float(row[6])
                col7 = float(row[7])
                col8 = float(row[8])
                col9 = float(row[9])
                Kamis_0sec_time.append(col6)
                Kamis_1sec_time.append(col7)
                Kamis_0sec_gurobi_time.append(col8)
                Kamis_1sec_gurobi_time.append(col9)
            except ValueError:
                continue  

if Kamis_0sec_values and Kamis_0sec_time:
    avg_Kamis_0sec_values = sum(Kamis_0sec_values) / len(Kamis_0sec_values)
    avg_Kamis_1sec_values = sum(Kamis_1sec_values) / len(Kamis_1sec_values)
    avg_Kamis_0sec_gurobi_values = sum(Kamis_0sec_gurobi_values) / len(Kamis_0sec_gurobi_values)
    avg_Kamis_1sec_gurobi_values = sum(Kamis_1sec_gurobi_values) / len(Kamis_1sec_gurobi_values)

    avg_Kamis_0sec_time = sum(Kamis_0sec_time) / len(Kamis_0sec_time)
    avg_Kamis_1sec_time = sum(Kamis_1sec_time) / len(Kamis_1sec_time)
    avg_Kamis_0sec_gurobi_time = sum(Kamis_0sec_gurobi_time) / len(Kamis_0sec_gurobi_time)
    avg_Kamis_1sec_gurobi_time = sum(Kamis_1sec_gurobi_time) / len(Kamis_1sec_gurobi_time)

    print(f"Kamis 0sec MIS/ time: {avg_Kamis_0sec_values}, {avg_Kamis_0sec_time}")
    print(f"Kamis 1sec MIS/ time: {avg_Kamis_1sec_values}, {avg_Kamis_1sec_time}")
    print(f"Kamis 0sec+Gurobi MIS/ time: {avg_Kamis_0sec_gurobi_values}, {avg_Kamis_0sec_gurobi_time}")
    print(f"Kamis 1sec+Gurobi MIS/ time: {avg_Kamis_1sec_gurobi_values}, {avg_Kamis_1sec_gurobi_time}")
else:
    print("No matching rows found.")
    
