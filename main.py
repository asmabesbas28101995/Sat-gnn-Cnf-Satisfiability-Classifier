import torch
from cnf_parser import parse_cnf
from graph_builder import build_graph
from model import SATGNN

def predict(cnf_path):
    num_vars, clauses = parse_cnf(cnf_path)
    x, edge_index = build_graph(num_vars, clauses)

    model = SATGNN()
    model.eval()

    with torch.no_grad():
        out = model(x, edge_index)
        pred = torch.argmax(out, dim=1).item()

    return "SAT" if pred == 1 else "UNSAT"


if __name__ == "__main__":
    result = predict("data/example.cnf")
    print("Prediction:", result)
