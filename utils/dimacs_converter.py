import os
import pickle
import networkx as nx

def assemble_dataset_from_gpickle(graph_directories, choose_n=None):
    dataset = []
    for graph_directory in graph_directories:
        graphs_found = 0
        for  filename in os.listdir(graph_directory):
            if choose_n and graphs_found >= choose_n:
                break
            if filename.endswith(".gpickle"):
                graphs_found+=1
                print(
                    "Graph ",
                    os.path.join(graph_directory, filename),
                    "is being imported ...",
                )
                with open(os.path.join(graph_directory, filename), "rb") as f:
                    G = pickle.load(f)
                    g =nx.relabel.convert_node_labels_to_integers(G, first_label=0)
                    os.makedirs(f"{graph_directory}_dimacs", exist_ok=True)
                    dimacs_filename = f"{graph_directory}_dimacs/{filename[:-8]}.dimacs"

                    with open(dimacs_filename, "w") as f:
                        # write the header
                        f.write("p EDGE {} {}\n".format(g.number_of_nodes(), g.number_of_edges()))
                        # now write all edges
                        for u, v in g.edges():
                            f.write("e {} {}\n".format(u+1, v+1))
    return

graph_directories = [
    ### ER 700-800 Graphs ###
    "/export2/curiekim/pCQO-mis-benchmark/graphs/er_700-800"
]
assemble_dataset_from_gpickle(graph_directories)
