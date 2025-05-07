from copy import deepcopy
import pandas
from datetime import datetime
import logging
import tqdm
import os
from lib.dataset_generation import assemble_dataset_from_gpickle
from solvers.pCQO_MIS import pCQOMIS_MGD
import sys
import os
import glob
import csv
#from solvers.CPSAT_MIS import CPSATMIS
# from solvers.Gurobi_MIS import GurobiMIS
from solvers.KaMIS import ReduMIS
# from solvers.previous_work_MIS_dNNs import DNNMIS
from solvers.Gurobi_MIS_warm import GurobiMIS_warm


os.environ["GUROBI_HOME"] = "/export2/curiekim/gurobi1200/linux64"
os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('GUROBI_HOME')}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["GRB_LICENSE_FILE"] = "/export2/curiekim/gurobi.lic"

logger = logging.getLogger(__name__)
logging.basicConfig(filename="benchmark.log", level=logging.INFO, style="{")

# Interval for saving solution checkpoints
SOLUTION_SAVE_INTERVAL = 1

#### GRAPH IMPORT ####
def get_dataset_name_from_path(graph_dir):
    parts = graph_dir.strip("/").split("/")
    N = parts[-2].split("_")[-1]  # "N_3000" → "3000"
    p = parts[-1]                 # "10"
    return f"KaMIS_er_{N}_0.{p}_0sec", f"KaMIS_er_{N}_0.{p}_1sec"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_ER_N_p_ReduMIS_Gurobi_w0.15.py <graph_directory>, e.g python benchmark_ER_N_p_ReduMIS_Gurobi_w0.15.py './graphs/er_graphs/N_3000/10' ")
        sys.exit(1)
    #### GRAPH IMPORT ####

    graph_dir = sys.argv[1]
    graph_directories = [graph_dir]

    # Assemble dataset from .gpickle files in the specified directories
    dataset = assemble_dataset_from_gpickle(graph_directories)
    intermediate_results_0sec, intermediate_results_1sec = get_dataset_name_from_path(graph_dir)
    #### SOLVER DESCRIPTION ####

    # Define solvers and their parameters
    base_solvers = [
        {"name": "ReduMIS", "class": ReduMIS, "params": {"time_limit":0, "dataset": intermediate_results_0sec}}, 
        {"name": "ReduMIS", "class": ReduMIS, "params": {"time_limit":1, "dataset": intermediate_results_1sec}}, 
        {"name": "Gurobi_warm", "class": GurobiMIS_warm, "params": {"time_limit": 30, "dataset": intermediate_results_0sec, "warm_sample_rate": 0.15}},
        {"name": "Gurobi_warm", "class": GurobiMIS_warm, "params": {"time_limit": 30, "dataset": intermediate_results_1sec, "warm_sample_rate": 0.15}},
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
                column_headings = [
                    solution["solution_method"] for solution in dataset_solutions
                ]

                # Collect sizes and times for each solution
                table_row.extend(
                    [solution["data"]["size"] for solution in dataset_solutions]
                )
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
        table.to_csv(
            f"zero_to_stage_{current_stage}_of_{total_stages}_total_stages_{datetime.now()}.csv"
        )


    #### BENCHMARKING CODE ####
    solutions = []
    path_solutions = []

    # Calculate total number of stages
    stage = 0
    stages = len(solvers) * len(dataset)

    # Iterate over each graph in the dataset
    for graph in tqdm.tqdm(dataset, desc=" Iterating Through Graphs", position=0):
        for solver in tqdm.tqdm(solvers, desc=" Iterating Solvers for Each Graph"):
            # To test localsearch, uncomment the following lines
            # initial_solver = CPSATMIS(graph["data"], {"time_limit": 5})
            # initial_solver.solve()

            # # convert graph mask to int
            # graph_mask = initial_solver.solution["graph_mask"].copy().astype(int)
            # # convert graph_mask to string with a space between each entry
            # graph_mask_str = " ".join(map(str, graph_mask))
            # print("Solutions found by CPSAT solver: ", initial_solver.solution["size"])

            # solver["params"]["initialization_vector"] = graph_mask_str


            #solver_instance = solver["class"](graph["data"], solver["params"])
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
                    "solution_method": f'{solver["name"]} {solver["params"]["dataset"]}',
                    "dataset_name": graph["name"],
                    "data": deepcopy(solver_instance.solution),
                    "time_taken": deepcopy(solver_instance.solution_time),
                }
                logging.info(
                    "CSV: %s, %s, %s",
                    graph["name"],
                    solution["data"]["size"],
                    solution["time_taken"],
                )
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
