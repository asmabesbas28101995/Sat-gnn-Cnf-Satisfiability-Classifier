# Sat-gnn-Cnf-Satisfiability-Classifier
SAT-GNN is a lightweight deep learning framework for SAT solving that classifies CNF formulas as SAT or UNSAT using Graph Neural Networks. Each CNF is converted into a bipartite graph of variables and clauses, and the model learns structural patterns from SATLIB 3-SAT benchmarks to predict satisfiability for unseen instances.

## 🧠 Methodology
1. **CNF Parsing** (DIMACS format)
2. **Graph Construction**
   - Variables → nodes  
   - Clauses → nodes  
   - Edges → literal relationships  
3. **Graph Neural Network (GNN)**
   - Message passing layers
   - Node embedding learning
4. **Graph Pooling**
   - Aggregation of node features
5. **Binary Classification**
   - Output: SAT / UNSAT

---

## 📊 Dataset
- SATLIB 3-SAT benchmark dataset
- Random 3-SAT instances near phase transition
- Balanced SAT and UNSAT samples

---

## ⚙️ Technologies Used
- Python 3
- PyTorch
- PyTorch Geometric
- NumPy
- SATLIB (DIMACS CNF format)

---

## 🚀 Project Structure

SAT-GNN/
│
├── data/                # CNF datasets (SATLIB)
├── cnf_parser.py       # DIMACS CNF parser
├── graph_builder.py    # CNF → graph conversion
├── model.py            # GNN architecture
├── train.py            # Model training script
├── main.py             # Inference (predict SAT/UNSAT)
├── requirements.txt
└── README.md


## Run
bash
python main.py





## We use 3-SAT random instances from SATLIB, particularly near the phase transition region.
