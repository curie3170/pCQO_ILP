from copy import deepcopy
import pickle
import pandas
import torch
from datetime import datetime
import logging
import tqdm
import sys
import os
import glob
import csv
from lib.dataset_generation import assemble_dataset_from_gpickle
from solvers.pCQO_MIS import pCQOMIS_MGD
from solvers.CPSAT_MIS import CPSATMIS
from solvers.Gurobi_MIS import GurobiMIS
from solvers.Gurobi_MIS_warm import GurobiMIS_warm
# from solvers.KaMIS import ReduMIS
# from solvers.previous_work_MIS_dNNs import DNNMIS

os.environ["GUROBI_HOME"] = "/export3/curiekim/gurobi1200/linux64"
os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('GUROBI_HOME')}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["GRB_LICENSE_FILE"] = "/export3/curiekim/gurobi.lic"


logger = logging.getLogger(__name__)
logging.basicConfig(filename='benchmark.log', level=logging.INFO, style="{")

# Interval for saving solution checkpoints
SOLUTION_SAVE_INTERVAL = 1

#### LOAD FAST TUNED HYPERPARAMETERS ####
def extract_params_from_csv(graph_dir):
    norm_graph_dir = os.path.normpath(graph_dir)
    if "graphs" not in norm_graph_dir:
        raise ValueError("Expected 'graphs' in the graph directory path.")

    relative_suffix = norm_graph_dir.split("graphs" + os.sep, 1)[1]  # er_graphs/N_3000/10
    csv_dir = os.path.normpath(os.path.join("..", "fobc", "fast_tune", relative_suffix)) # ../fobc/er_graphs/N_3000/10

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


def get_dataset_name_from_path(graph_dir, lr, mom, gamma, gamma_prime):
    parts = graph_dir.strip("/").split("/")
    N = parts[-2].split("_")[-1]  # "N_3000" → "3000"
    p = parts[-1]                 # "10"
    return f"fastpCQO_er_{N}_0.{p}_133200steps_{lr}_{mom}_{gamma}_{gamma_prime}", f"ER_{N}_0.{p}"
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_ER_N_p_Gurobi_w0.15.py <graph_directory>, e.g python benchmark_ER_N_p_Gurobi_w0.15.py './graphs/er_graphs/N_3000/10' ")
        sys.exit(1)

    #### GRAPH IMPORT ####

    graph_dir = sys.argv[1]
    graph_directories = [graph_dir]

    # Assemble dataset from .gpickle files in the specified directories
    dataset = assemble_dataset_from_gpickle(graph_directories)

    # set fast tuned hyperparameters
    lr, mom, gamma, gamma_prime = extract_params_from_csv(graph_dir)
    intermediate_results, graph_N_p = get_dataset_name_from_path(graph_dir, lr, mom, gamma, gamma_prime)
    
    #### SOLVER DESCRIPTION ####

    # Define solvers and their parameters
    base_solvers = [
        {
            "name": f"Fast tuned pCQO_MIS {graph_N_p}",
            "class": pCQOMIS_MGD,
            "params": {
                "set_of_params": [[lr, mom, gamma, gamma_prime]],
                # "learning_rate": 0.000009,
                # "momentum": 0.9,
                "number_of_steps": 133200,#225000, 
                # "gamma": 350,
                # "gamma_prime": 7,
                "batch_size": 256,
                "std": 2.25,
                "threshold": 0.00,
                "steps_per_batch": 450,
                "output_interval": 133202, #225002,
                "value_initializer": "degree",
                "checkpoints": [450] + list(range(4500, 133202, 4500)), #225002
                #"time_limit": 30, #don't save checkpoints
                "dataset": intermediate_results
            },
        },
        {"name": "Gurobi_warm 450", "class": GurobiMIS_warm, "params": {"time_limit": 30, "dataset": intermediate_results, "iteration": 450, "warm_sample_rate": 0.15}},
        {"name": "Gurobi_warm 133200", "class": GurobiMIS_warm, "params": {"time_limit": 30, "dataset": intermediate_results, "iteration": 133200, "warm_sample_rate": 0.15}}, #pcqo30sec value
    ]

    solvers = base_solvers

    ## Grid Search (Commented Out)
    # Uncomment and configure the following section for hyperparameter tuning

    # solvers = []
    # for solver in base_solvers:
    #     for learning_rate in [0.001]:
    #         for momentum in [0.5]:
    #             for gamma_gamma_prime in [(500, 1),]:
    #                 for batch_size in [256]:
    #                     for terms in ["three"]:
    #                         modified_solver = deepcopy(solver)
    #                         modified_solver["name"] = (
    #                             f"{modified_solver['name']} batch_size={batch_size}, learning_rate={learning_rate}, momentum={momentum}, gamma={gamma_gamma_prime[0]}, gamma_prime={gamma_gamma_prime[1]},  terms={terms}"
    #                         )
    #                         modified_solver["params"]["learning_rate"] = learning_rate
    #                         modified_solver["params"]["momentum"] = momentum
    #                         modified_solver["params"]["gamma"] = gamma_gamma_prime[0]
    #                         modified_solver["params"]["gamma_prime"] = gamma_gamma_prime[1]
    #                         modified_solver["params"]["number_of_terms"] = terms
    #                         modified_solver["params"]["batch_size"] = batch_size
    #                         solvers.append(modified_solver)



    #### SOLUTION OUTPUT FUNCTION ####
    def table_output(solutions, datasets, current_stage, total_stages):
        """
        Saves the solutions to a CSV file.

        Args:
            solutions (list): List of solution dictionaries.
            datasets (list): List of dataset dictionaries.
            current_stage (int): Current stage in the benchmarking process.
            total_stages (int): Total number of stages in the benchmarking process.
        """
        # Create a mapping of dataset names to indices
        dataset_index = {dataset["name"]: index for index, dataset in enumerate(datasets)}
        datasets_solutions = [[] for _ in range(len(datasets))]
        table_data = []

        # Organize solutions by dataset
        for solution in solutions:
            dataset_idx = dataset_index[solution["dataset_name"]]
            datasets_solutions[dataset_idx].append(solution)

        # Prepare data for the output table
        for dataset_solutions in datasets_solutions:
            if dataset_solutions:
                table_row = [dataset_solutions[0]["dataset_name"]]
                column_headings = [solution["solution_method"] for solution in dataset_solutions]

                # Collect sizes and times for each solution
                table_row.extend([solution["data"]["size"] for solution in dataset_solutions])
                # Uncomment to include steps to solution size if available
                # table_row.extend([solution['data']['steps_to_best_MIS'] for solution in dataset_solutions])
                table_row.extend([solution["time_taken"] for solution in dataset_solutions])

                table_data.append(table_row)

        # Generate headers for the CSV file
        table_headers = ["Dataset Name"]
        table_headers.extend([heading + " Solution Size" for heading in column_headings])
        # Uncomment to include headers for steps to solution size if available
        # table_headers.extend([heading + " # Steps to Solution Size" for heading in column_headings])
        table_headers.extend([heading + " Solution Time" for heading in column_headings])

        # Save the data to a CSV file
        table = pandas.DataFrame(table_data, columns=table_headers)
        table.to_csv(f"zero_to_stage_{current_stage}_of_{total_stages}_total_stages_{datetime.now()}.csv")

    #### BENCHMARKING CODE ####
    solutions = []
    path_solutions = []

    # Calculate total number of stages
    stage = 0
    stages = len(solvers) * len(dataset)

    # Iterate over each graph in the dataset
    for graph in tqdm.tqdm(dataset, desc=" Iterating Through Graphs", position=0):
        for solver in tqdm.tqdm(solvers, desc=" Iterating Solvers for Each Graph"):
            solver_instance = solver["class"](graph["data"], graph["name"], solver["params"])

            # Solve the problem using the current solver
            solver_instance.solve()
            if hasattr(solver_instance, "solutions") and len(solver_instance.solutions) > 0:
                for solution in solver_instance.solutions:
                    pretty_solution = {
                        "solution_method": f"{solver['name']} at step {solution['number_of_steps']}",
                        "dataset_name": graph["name"],
                        "data": deepcopy(solution),
                        "time_taken": deepcopy(solution["time"]),
                    }
                    solutions.append(pretty_solution)
            else:
                solution = {
                    "solution_method": f'{solver["name"]} warm start at step {solver["params"]["iteration"]} with random sample rate of {solver["params"]["warm_sample_rate"]} in dataset {solver["params"]["dataset"]}',
                    "dataset_name": graph["name"],
                    "data": deepcopy(solver_instance.solution),
                    "time_taken": deepcopy(solver_instance.solution_time),
                }
                logging.info("CSV: %s, %s, %s", graph['name'], solution['data']['size'], solution['time_taken'])
                solutions.append(solution)
            del solver_instance

            # Update progress and save checkpoint if necessary
            stage += 1
            logger.info("Completed %s / %s", stage, stages)


            if stage % (SOLUTION_SAVE_INTERVAL * len(solvers)) == 0:
                logger.info("Now saving a checkpoint.")
                table_output(solutions, dataset, stage, stages)

    # Save final results
    logger.info("Now saving final results.")
    table_output(solutions, dataset, stage, stages)
