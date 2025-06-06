import networkx as nx
from gurobipy import Model, GRB, quicksum
from lib.Solver import Solver
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
class GurobiMIS_warm(Solver):
    """
    A solver class for finding the Maximum Independent Set (MIS) of a graph using the Gurobi optimization solver.

    Parameters:
        G (networkx.Graph): The graph on which the MIS problem will be solved.
        params (dict): Dictionary containing solver parameters:
            - time_limit (int, optional): Time limit (in seconds) for the solver to run. Defaults to None.
    """
    def __init__(self, G, G_name, params):
        """
        Initializes the GurobiMIS solver with the given graph and parameters.

        Args:
            G (networkx.Graph): The graph to solve the MIS problem on.
            params (dict): Parameters for the solver, including optional time_limit.
        """
        self.G = G
        self.graph_name = G_name
        self.time_limit = params.get("time_limit", None)
        self.dataset = params.get("dataset", None)
        self.iteration = params.get("iteration", 0)
        self.warm_sample_rate = params.get("warm_sample_rate", 1.0)
        # self.confidence_th = params.get("confidence_th", 0)
        self.solution = {}
        self.model = None
        self.solution_time = None  # Initialize solution_time
        self.paths = []
        self.times = []

    def data_cb(self, model, where):
        """
        Callback function for Gurobi's MIP solver.

        This function is called during the optimization process to monitor the progress.
        It records the best objective value and corresponding runtime at each callback.

        Args:
            model (gurobipy.Model): The Gurobi model being optimized.
            where (int): The location in the optimization process where the callback is invoked.
        """
        if where == GRB.Callback.MIP:
            cur_obj = model.cbGet(GRB.Callback.MIP_OBJBST)

            self._stop_timer()
            if len(self.paths) == 0 or self.paths[-1] < cur_obj:
                self.times.append(self.solution_time)
                self.paths.append(cur_obj)

        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            print(f"New incumbent solution found at {runtime:.2f} seconds with objective {obj:.6f}")
            solution = model.cbGetSolution(model.getVars())
            indices = [i for i, x in enumerate(solution) if x == 1.0]
            print(indices)
    def solve(self):
        """
        Executes the Gurobi solver to find the Maximum Independent Set (MIS) of the graph.

        The method performs the following steps:
        1. Creates and configures a new Gurobi model for the MIS problem.
        2. Defines binary variables for each node in the graph.
        3. Adds constraints to ensure no two adjacent nodes are both in the independent set.
        4. Sets the objective function to maximize the number of selected nodes.
        5. Optimizes the model and records the solution time.
        6. Extracts and prints the solution if the model finds an optimal or feasible solution.
        7. Prints the progress paths and times.

        Outputs:
            - self.solution (dict): Contains the results of the MIS computation:
                - graph_mask (list of int): List of 0s and 1s where 1s denote nodes in the MIS.
                - size (int): Size of the MIS.
        """
        # Create a new Gurobi model
        self.model = Model("Maximum_Independent_Set")
        #self.model.params.MIPFocus=1
        # Set the time limit if specified
        if self.time_limit is not None:
            self.model.setParam("TimeLimit", self.time_limit)

        # Create a binary variable for each node
        node_vars = {
            node: self.model.addVar(vtype=GRB.BINARY, name=f"node_{node}")
            for node in self.G.nodes
        }

        # Add constraints: no two adjacent nodes can both be in the independent set
        for u, v in self.G.edges:
            if u != v:  # avoid adding a constraint for self-loops, if they exist
                self.model.addConstr(node_vars[u] + node_vars[v] <= 1, f"edge_{u}_{v}")

        # Initialize with intermediate values
        # if (self.dataset is not None) and (self.iteration is not 0) and (self.confidence_th > 0):
        #     df = pd.read_csv(f'./intermediate_results/{self.dataset}/{self.graph_name}_{self.iteration}_confidence{self.confidence_th}.csv')
        if (self.dataset is not None) and (self.iteration is not 0):
            csv_path = f'./intermediate_results/{self.dataset}/{self.graph_name}_{self.iteration}.csv'
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                df = pd.read_csv(csv_path)
            else:
                logger.warning("CSV not found or empty: %s", csv_path)
                df = pd.DataFrame()
        elif (self.dataset is not None) and (self.iteration is 0):
            df = pd.read_csv(f'./intermediate_results/{self.dataset}/{self.graph_name}.csv')
        else:
            df = pd.read_csv(f'./intermediate_results/{self.graph_name}.csv')
        warm_start = df.values
        sample_size = int(len(warm_start) * self.warm_sample_rate)
        warm_start = warm_start[np.random.choice(len(warm_start), size=sample_size, replace=False)]
        for node in warm_start:
            node_vars[int(node)].Start = 1 
        self.model.update()
        
        # Set the objective: maximize the sum of the selected nodes
        self.model.setObjective(
            quicksum(node_vars[node] for node in self.G.nodes), GRB.MAXIMIZE
        )

        # Optimize the model
        self._start_timer()
        self.model.write("./model_warmfull.lp")
        self.model.optimize(callback=self.data_cb)
        self.solution_time = self.model.Runtime

        # Check if a valid solution exists
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            self.solution["graph_mask"] = [
                int(node_vars[node].X) for node in self.G.nodes
            ]
            self.solution["size"] = sum(self.solution["graph_mask"])
            print(f"Maximum Independent Set size: {self.solution['size']}")
        else:
            print("No valid solution found.")
            self.solution["graph_mask"] = []
            self.solution["size"] = 0

        print(self.paths, self.times)

        # Optional: Output the variables if the solution was found
        if self.model.status == GRB.OPTIMAL:
            print("Nodes in the independent set:")
            for node in self.G.nodes:
                if node_vars[node].X > 0.5:  # effectively checking if the variable is 1
                    print(node)

if __name__ == "__main__":
    # Create a simple example graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

    # Set parameters for the GurobiMIS solver
    params = {"time_limit": 10}  # 10 seconds time limit

    # Initialize and solve the MIS problem
    solver = GurobiMIS(G, params)
    solver.solve()
