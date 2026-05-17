import torch

def build_graph(num_vars, clauses):
    edges = []

    for ci, clause in enumerate(clauses):
        clause_id = num_vars + ci

        for lit in clause:
            var = abs(lit) - 1
            edges.append([var, clause_id])
            edges.append([clause_id, var])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = num_vars + len(clauses)

    x = torch.ones((num_nodes, 8))

    return x, edge_index
