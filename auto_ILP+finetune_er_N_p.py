import csv
import os
import glob
import sys
from pathlib import Path
import subprocess
def extract_params_from_csv(graph_dir):
    norm_graph_dir = os.path.normpath(graph_dir)
    if "graphs" not in norm_graph_dir:
        raise ValueError("Expected 'graphs' in the graph directory path.")

    #relative_suffix = norm_graph_dir.split("graphs" + os.sep, 1)[1]
    csv_dir = os.path.normpath(os.path.join("..", "fobc", "fast_tune", norm_graph_dir))
    print(csv_dir)
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*_repeated_fixed_sample.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    csv_path = csv_files[0]
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        lr = float(row["LearningRate"])
        mom = float(row["Momentum"])
        gamma = int(row["Gamma"])
        gamma_prime = int(row["GammaPrime"])

    return lr, mom, gamma, gamma_prime

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_ILP+finetune_er_N_p.py <subdirectory under fobc/graphs/>, auto_ILP+finetune_er_N_p.py 'er_graphs/N_3000/30' ")
        sys.exit(1)
    
    graph_dir = sys.argv[1]
    lr, mom, gamma, gamma_prime = extract_params_from_csv(graph_dir)

    cmd = [
        "bash",
        "script_ILP+finetune_er_700800_lr_mom_gamma_gammaprime.sh",
        str(graph_dir),
        str(lr),
        str(mom),
        str(gamma),
        str(gamma_prime),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)