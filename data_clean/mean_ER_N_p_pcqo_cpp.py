import csv
import re
import glob
import os
result_dir = "/export3/curiekim/pCQO-mis-benchmark/meaningful_results"
pattern = os.path.join(result_dir, "*_fasttune(repeated_fixed_CPP)500step_30s_Gurobi.csv")
all_result_files = glob.glob(pattern)
print(f"Found {len(all_result_files)} result files.\n")


for filename in all_result_files:
    pcqo500_gurobi_values = []
    pcqo30s_gurobi_values = []
    pcqo500_gurobi_time = []
    pcqo30s_gurobi_time = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            pcqo500_gurobi_values.append(float(row[2]))  
            pcqo30s_gurobi_values.append(float(row[3])) 
            pcqo500_gurobi_time.append(float(row[4]))  
            pcqo30s_gurobi_time.append(float(row[5]))  

    avg_pcqo500_gurobi_mis = sum(pcqo500_gurobi_values) / len(pcqo500_gurobi_values)
    avg_pcqo30s_gurobi_mis = sum(pcqo30s_gurobi_values) / len(pcqo30s_gurobi_values)
    avg_pcqo500_gurobi_time = sum(pcqo500_gurobi_time) / len(pcqo500_gurobi_time)
    avg_pcqo30s_gurobi_time = sum(pcqo30s_gurobi_time) / len(pcqo30s_gurobi_time)
    
    
    #print(f"Processing: {filename}")
    # 1. 정규표현식으로 N과 p 추출
    match = re.search(r'ER(\d+)\.(\d+)', filename)
    if match:
        N = match.group(1)          # '3000'
        p_raw = match.group(2)      # '70'
        p_formatted = f"0.{p_raw}"  # '0.70'
    else:
        raise ValueError("Could not extract N and p from filename.")
    print(f"Average pCQO500step+ Gurobi MIS/ time in ER_{N}_{p_raw}: {avg_pcqo500_gurobi_mis}, {avg_pcqo500_gurobi_time}")
    print(f"Average pCQO30s+ Gurobi MIS/ time in ER_{N}_{p_raw}: {avg_pcqo30s_gurobi_mis}, {avg_pcqo30s_gurobi_time}")
    # 2. glob 패턴 경로 생성
    pattern_500steps = f"/export3/curiekim/pCQO-mis-benchmark/intermediate_results/er_graphs_N_{N}_{p_raw}_*/ER_{N}_{p_formatted}_*_500.csv"

    # 3. 실제 경로 찾기
    files_500steps = glob.glob(pattern_500steps)

    if files_500steps:
        mis = 0
        for i in range(len(files_500steps)):
            with open(files_500steps[i], "r") as f:
                row_count = sum(1 for _ in f)
                mis += row_count
                #print(f"pCQO 500step MIS: {row_count}")
        print(f"Average pCQO 500step MIS in ER_{N}_{p_raw}: {mis/ len(files_500steps)}")
    else:
        print("No matching file found.")

    if files_500steps:
        mis = 0
        for i in range(len(files_500steps)):
            basename = os.path.basename(files_500steps[i])
            files_30sec = re.sub(r'_\d+\.csv$', '.csv', basename)
            path_30sec = os.path.join(os.path.dirname(files_500steps[i]), files_30sec)

            with open(path_30sec, "r") as f:
                row_count = sum(1 for _ in f)
                mis += row_count
                #print(f"pCQO 30s MIS: {row_count}")
        print(f"Average pCQO 30s MIS in ER_{N}_{p_raw}: {mis/ len(files_500steps)}")
    else:
        print("No matching file found.")
        


# Found 35 result files.

# Average pCQO500step+ Gurobi MIS/ time in ER_3000_70: 8.3, 30.506493544578554
# Average pCQO30s+ Gurobi MIS/ time in ER_3000_70: 8.8, 30.510289001464844
# Average pCQO 500step MIS in ER_3000_70: 9.0
# Average pCQO 30s MIS in ER_3000_70: 9.0
# Average pCQO500step+ Gurobi MIS/ time in ER_3000_60: 8.9, 30.190552473068237
# Average pCQO30s+ Gurobi MIS/ time in ER_3000_60: 9.0, 30.185246753692628
# Average pCQO 500step MIS in ER_3000_60: 11.2
# Average pCQO 30s MIS in ER_3000_60: 11.5
# Average pCQO500step+ Gurobi MIS/ time in ER_3000_50: 12.2, 30.154945397377013
# Average pCQO30s+ Gurobi MIS/ time in ER_3000_50: 12.7, 30.152213072776796
# Average pCQO 500step MIS in ER_3000_50: 14.2
# Average pCQO 30s MIS in ER_3000_50: 14.3
# Average pCQO500step+ Gurobi MIS/ time in ER_3000_40: 16.6, 30.117430925369263
# Average pCQO30s+ Gurobi MIS/ time in ER_3000_40: 16.5, 30.11750810146332
# Average pCQO 500step MIS in ER_3000_40: 18.3
# Average pCQO 30s MIS in ER_3000_40: 18.5
# Average pCQO500step+ Gurobi MIS/ time in ER_3000_30: 22.5, 30.09489095211029
# Average pCQO30s+ Gurobi MIS/ time in ER_3000_30: 21.2, 30.093543934822083
# Average pCQO 500step MIS in ER_3000_30: 25.0
# Average pCQO 30s MIS in ER_3000_30: 25.2
# Average pCQO500step+ Gurobi MIS/ time in ER_3000_20: 29.7, 30.510670137405395
# Average pCQO30s+ Gurobi MIS/ time in ER_3000_20: 32.4, 30.535663223266603
# Average pCQO 500step MIS in ER_3000_20: 36.9
# Average pCQO 30s MIS in ER_3000_20: 39.8
# Average pCQO500step+ Gurobi MIS/ time in ER_3000_10: 58.4, 30.548507046699523
# Average pCQO30s+ Gurobi MIS/ time in ER_3000_10: 56.7, 30.526015996932983
# Average pCQO 500step MIS in ER_3000_10: 67.6
# Average pCQO 30s MIS in ER_3000_10: 76.2
# Average pCQO500step+ Gurobi MIS/ time in ER_2500_70: 8.7, 30.301096057891847
# Average pCQO30s+ Gurobi MIS/ time in ER_2500_70: 8.4, 30.435050821304323
# Average pCQO 500step MIS in ER_2500_70: 9.2
# Average pCQO 30s MIS in ER_2500_70: 9.2
# Average pCQO500step+ Gurobi MIS/ time in ER_2500_60: 9.1, 30.766016817092897
# Average pCQO30s+ Gurobi MIS/ time in ER_2500_60: 8.7, 30.784797739982604
# Average pCQO 500step MIS in ER_2500_60: 11.1
# Average pCQO 30s MIS in ER_2500_60: 11.1
# Average pCQO500step+ Gurobi MIS/ time in ER_2500_50: 11.1, 30.804328870773315
# Average pCQO30s+ Gurobi MIS/ time in ER_2500_50: 11.9, 30.79333403110504
# Average pCQO 500step MIS in ER_2500_50: 13.8
# Average pCQO 30s MIS in ER_2500_50: 13.8
# Average pCQO500step+ Gurobi MIS/ time in ER_2500_40: 14.7, 30.69217367172241
# Average pCQO30s+ Gurobi MIS/ time in ER_2500_40: 14.3, 30.690511918067934
# Average pCQO 500step MIS in ER_2500_40: 18.2
# Average pCQO 30s MIS in ER_2500_40: 18.6
# Average pCQO500step+ Gurobi MIS/ time in ER_2500_30: 18.9, 30.604980421066283
# Average pCQO30s+ Gurobi MIS/ time in ER_2500_30: 19.2, 30.578488612174986
# Average pCQO 500step MIS in ER_2500_30: 24.2
# Average pCQO 30s MIS in ER_2500_30: 24.7
# Average pCQO500step+ Gurobi MIS/ time in ER_2500_20: 29.8, 30.593412613868715
# Average pCQO30s+ Gurobi MIS/ time in ER_2500_20: 29.3, 30.580734610557556
# Average pCQO 500step MIS in ER_2500_20: 35.8
# Average pCQO 30s MIS in ER_2500_20: 39.2
# Average pCQO500step+ Gurobi MIS/ time in ER_2500_10: 60.4, 30.110608792304994
# Average pCQO30s+ Gurobi MIS/ time in ER_2500_10: 61.9, 30.098880290985107
# Average pCQO 500step MIS in ER_2500_10: 68.1
# Average pCQO 30s MIS in ER_2500_10: 69.8
# Average pCQO500step+ Gurobi MIS/ time in ER_2000_70: 8.8, 30.098096776008607
# Average pCQO30s+ Gurobi MIS/ time in ER_2000_70: 9.0, 30.09804763793945
# Average pCQO 500step MIS in ER_2000_70: 9.1
# Average pCQO 30s MIS in ER_2000_70: 9.1
# Average pCQO500step+ Gurobi MIS/ time in ER_2000_60: 10.2, 30.081323409080504
# Average pCQO30s+ Gurobi MIS/ time in ER_2000_60: 10.5, 30.08213860988617
# Average pCQO 500step MIS in ER_2000_60: 11.2
# Average pCQO 30s MIS in ER_2000_60: 11.5
# Average pCQO500step+ Gurobi MIS/ time in ER_2000_50: 11.7, 30.075095963478088
# Average pCQO30s+ Gurobi MIS/ time in ER_2000_50: 12.3, 30.073731875419618
# Average pCQO 500step MIS in ER_2000_50: 14.2
# Average pCQO 30s MIS in ER_2000_50: 14.4
# Average pCQO500step+ Gurobi MIS/ time in ER_2000_40: 16.6, 30.062287878990173
# Average pCQO30s+ Gurobi MIS/ time in ER_2000_40: 16.7, 30.06139733791351
# Average pCQO 500step MIS in ER_2000_40: 17.8
# Average pCQO 30s MIS in ER_2000_40: 18.1
# Average pCQO500step+ Gurobi MIS/ time in ER_2000_30: 23.2, 30.05073299407959
# Average pCQO30s+ Gurobi MIS/ time in ER_2000_30: 22.8, 30.04967782497406
# Average pCQO 500step MIS in ER_2000_30: 23.4
# Average pCQO 30s MIS in ER_2000_30: 24.6
# Average pCQO500step+ Gurobi MIS/ time in ER_2000_20: 33.9, 30.050021505355836
# Average pCQO30s+ Gurobi MIS/ time in ER_2000_20: 34.6, 30.04037549495697
# Average pCQO 500step MIS in ER_2000_20: 35.2
# Average pCQO 30s MIS in ER_2000_20: 39.3
# Average pCQO500step+ Gurobi MIS/ time in ER_2000_10: 64.7, 30.037832307815552
# Average pCQO30s+ Gurobi MIS/ time in ER_2000_10: 64.4, 30.047645449638367
# Average pCQO 500step MIS in ER_2000_10: 66.3
# Average pCQO 30s MIS in ER_2000_10: 69.5
# Average pCQO500step+ Gurobi MIS/ time in ER_1500_70: 8.9, 30.069364953041077
# Average pCQO30s+ Gurobi MIS/ time in ER_1500_70: 8.8, 30.067295360565186
# Average pCQO 500step MIS in ER_1500_70: 8.8
# Average pCQO 30s MIS in ER_1500_70: 9.0
# Average pCQO500step+ Gurobi MIS/ time in ER_1500_60: 10.6, 30.058181381225587
# Average pCQO30s+ Gurobi MIS/ time in ER_1500_60: 10.8, 30.05976219177246
# Average pCQO 500step MIS in ER_1500_60: 11.0
# Average pCQO 30s MIS in ER_1500_60: 11.0
# Average pCQO500step+ Gurobi MIS/ time in ER_1500_50: 12.5, 30.049752116203308
# Average pCQO30s+ Gurobi MIS/ time in ER_1500_50: 12.3, 30.053061962127686
# Average pCQO 500step MIS in ER_1500_50: 13.4
# Average pCQO 30s MIS in ER_1500_50: 13.8
# Average pCQO500step+ Gurobi MIS/ time in ER_1500_40: 17.0, 30.045085000991822
# Average pCQO30s+ Gurobi MIS/ time in ER_1500_40: 17.1, 30.04786260128021
# Average pCQO 500step MIS in ER_1500_40: 17.0
# Average pCQO 30s MIS in ER_1500_40: 17.2
# Average pCQO500step+ Gurobi MIS/ time in ER_1500_30: 23.0, 30.040655493736267
# Average pCQO30s+ Gurobi MIS/ time in ER_1500_30: 22.9, 30.03709168434143
# Average pCQO 500step MIS in ER_1500_30: 22.2
# Average pCQO 30s MIS in ER_1500_30: 23.5
# Average pCQO500step+ Gurobi MIS/ time in ER_1500_20: 33.6, 30.032502508163454
# Average pCQO30s+ Gurobi MIS/ time in ER_1500_20: 35.6, 30.0533807516098
# Average pCQO 500step MIS in ER_1500_20: 33.9
# Average pCQO 30s MIS in ER_1500_20: 39.6
# Average pCQO500step+ Gurobi MIS/ time in ER_1500_10: 63.6, 30.101883721351623
# Average pCQO30s+ Gurobi MIS/ time in ER_1500_10: 63.0, 30.10082652568817
# Average pCQO 500step MIS in ER_1500_10: 63.5
# Average pCQO 30s MIS in ER_1500_10: 66.1
# Average pCQO500step+ Gurobi MIS/ time in ER_1000_70: 8.8, 30.03272376060486
# Average pCQO30s+ Gurobi MIS/ time in ER_1000_70: 8.8, 30.032282328605653
# Average pCQO 500step MIS in ER_1000_70: 8.5
# Average pCQO 30s MIS in ER_1000_70: 8.9
# Average pCQO500step+ Gurobi MIS/ time in ER_1000_60: 10.8, 30.040893626213073
# Average pCQO30s+ Gurobi MIS/ time in ER_1000_60: 11.0, 30.042830610275267
# Average pCQO 500step MIS in ER_1000_60: 10.5
# Average pCQO 30s MIS in ER_1000_60: 11.3
# Average pCQO500step+ Gurobi MIS/ time in ER_1000_50: 13.1, 30.03302617073059
# Average pCQO30s+ Gurobi MIS/ time in ER_1000_50: 12.8, 30.036267399787903
# Average pCQO 500step MIS in ER_1000_50: 13.2
# Average pCQO 30s MIS in ER_1000_50: 13.7
# Average pCQO500step+ Gurobi MIS/ time in ER_1000_40: 17.2, 30.027103424072266
# Average pCQO30s+ Gurobi MIS/ time in ER_1000_40: 16.9, 30.029832577705385
# Average pCQO 500step MIS in ER_1000_40: 16.6
# Average pCQO 30s MIS in ER_1000_40: 18.5
# Average pCQO500step+ Gurobi MIS/ time in ER_1000_30: 22.7, 30.028469586372374
# Average pCQO30s+ Gurobi MIS/ time in ER_1000_30: 22.8, 30.02845695018768
# Average pCQO 500step MIS in ER_1000_30: 22.0
# Average pCQO 30s MIS in ER_1000_30: 24.0
# Average pCQO500step+ Gurobi MIS/ time in ER_1000_20: 33.7, 30.028521013259887
# Average pCQO30s+ Gurobi MIS/ time in ER_1000_20: 35.3, 30.02605504989624
# Average pCQO 500step MIS in ER_1000_20: 32.3
# Average pCQO 30s MIS in ER_1000_20: 37.5
# Average pCQO500step+ Gurobi MIS/ time in ER_1000_10: 59.5, 30.025003695487975
# Average pCQO30s+ Gurobi MIS/ time in ER_1000_10: 62.5, 30.028381490707396
# Average pCQO 500step MIS in ER_1000_10: 57.9
# Average pCQO 30s MIS in ER_1000_10: 67.1