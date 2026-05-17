def parse_cnf(file_path):
    clauses = []
    num_vars = 0

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('c') or line.startswith('%'):
                continue
            if line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
            else:
                clause = list(map(int, line.strip().split()[:-1]))
                if clause:
                    clauses.append(clause)

    return num_vars, clauses
