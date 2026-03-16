import sys
import subprocess
import time
import re
import random
import numpy as np


from reductions.blp2dgp import reduce_blp_2_dgp

### GLOBAL CONSTANTS
myEps = 1e-2
minVal = -10
maxVal = 10
wrongPerturbRange = 3


def generateSparseIntegerMatrix(m, n, s, min_val=minVal, max_val=maxVal):
    # Generate a random m x n integer matrix with values in [min_val, max_val]
    A = np.random.randint(min_val, max_val+1, size=(m, n))
    # Zero out some entries according to sparsity s (s is the density threshold)
    mask = np.random.random(size=(m, n))
    A[mask > s] = 0
    # Ensure no row is entirely zero: if so, force one entry nonzero.
    for i in range(m):
        if abs(np.sum(A[i, :])) < myEps:
            newindex = np.random.randint(0, n)
            A[i, newindex] = np.random.randint(min_val, max_val+1)
    return A

def generate_dat_graph(filename, n=100, m=500, weight_min=1, weight_max=5):
    edges = set()
    while len(edges) < m:
        i = random.randint(1, n)
        j = random.randint(1, n)
        if i != j:
            edges.add(tuple(sorted((i, j))))

    edges = list(edges)

    with open(filename, "w") as f:
        # header
        f.write("# DGP instance in .dat format for dgp.mod\n")
        f.write("# this graph is a DGP encoding of a random instance\n\n")

        # edge list
        f.write("param : E : c I :=\n")

        for (i, j) in edges:
            w = random.randint(weight_min, weight_max)
            f.write(f"  {i} {j}  {w:.3f} 0\n")

        f.write(";\n")

def generate_feasible_dat_graph(filename, n=8, m=10, A=10):
    #randomly position n points on the line at integer coordinates between 0 and A
    points = random.sample(range(max(A, 2*n)), n)
    points.sort()
    edges = set()
    while len(edges) < m:
        i = random.randint(1, n)
        j = random.randint(1, n)
        if i != j:
            if i < j:
                edges.add(tuple((i, j, abs(points[i-1] - points[j-1]))))
            else:
                edges.add(tuple((j, i, abs(points[i-1] - points[j-1]))))

    with open(filename, "w") as f:
        # header
        f.write("# DGP instance in .dat format for dgp.mod\n")
        f.write("# this graph is a DGP encoding of a random instance\n\n")

        # edge list
        f.write("param : E : c I :=\n")

        for (i, j, w) in edges:
            #the weight is not random here as it comes from the distance between the points in the 1D embedding
            f.write(f"  {i} {j}  {w:.3f} 0\n")

        f.write(";\n")

def generate_instance(n, m, s, feasType):
    """
    Generates a BLP instance and returns an OPB string.
    The matrix A is generated sparsely and then b is set based on the type:
      - "feas": b = A*x where x is a random binary vector.
      - "rnd": b is random.
      - "infeas": b[i] is set to min(A[i,:]) - 1 (forcing infeasibility).
      - "perturb": start with a feasible instance then perturb one random column.
    """
    A = generateSparseIntegerMatrix(m, n, s)
    b = np.zeros(m, dtype=int)
    perturb = None
    perturbColIndex = -1

    if feasType == "feas":
        xrnd = np.random.randint(0, 2, size=n)
        b = A.dot(xrnd)
    elif feasType == "rnd":
        b = np.random.randint(minVal, maxVal+1, size=m)
    elif feasType == "infeas":
        for i in range(m):
            # Use the minimum nonzero entry (or minimum overall) and subtract 1.
            b[i] = int(np.min(A[i, :]) - 1)
    elif feasType == "perturb":
        xrnd = np.random.randint(0, 2, size=n)
        b = A.dot(xrnd)
        perturb = np.random.randint(-wrongPerturbRange, wrongPerturbRange, size=m)
        perturbColIndex = np.random.randint(0, n)
        A[:, perturbColIndex] += perturb
    # Build the OPB instance string.
    opb_lines = []
    for i in range(m):
        terms = []
        for j in range(n):
            if A[i, j] > myEps:
                terms.append("+{}*x{}".format(int(A[i,j]), j+1))
            elif A[i, j] < -myEps:
                terms.append("{}*x{}".format(int(A[i,j]), j+1))
        # We use "<=" for each constraint.
        opb_lines.append(" ".join(terms) + " <= {};".format(int(b[i])))
    opb_str = "\n".join(opb_lines)
    return opb_str

def writeDat(G, dgpf, opbf):
    (V,E, VL, nlits) = G
    n = max(V)
    with open(dgpf, "w") as f:
        print("# DGP instance in .dat format for dgp.mod", file=f)
        print("param : E : c I :=", file=f)

        for el in E:
            print(el[0], el[1], el[2], el[3], VL[el[0]], VL[el[1]])
            print("  {} {}  {:.3f} {}  # [{},{}]".format(el[0],el[1],el[2],el[3],VL[el[0]],VL[el[1]]), file=f)
        print(";", file=f)
        #print("# vertex map", file=f)
        #for i in V:
        #    print("# vertex {} is {}".format(i, VL[i]), file=f)
        print(f"Successfully writen DGP instance to {dgpf}")
    return

def readOpb(opbf):
    b = {}    # RHS
    rel = {}  # Relations
    Ax = {}   # LHS
    var_ids = set()
    with open(opbf, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line[0] == '#':
                continue
            l = line.split()
            if l[-1] == ';':
                b[i] = int(l[-2])
                rel[i] = l[-3]
                l = l[:-3]
            else:
                b[i] = int(l[-1][:-1])
                rel[i] = l[-2]
                l = l[:-2]
            Ax[i] = {}
            for term in l:
                coeff, var = term.split('*')
                j = int(var[1:])  # 'x3' → 3
                coeff = int(coeff)
                Ax[i][j] = coeff
                var_ids.add(abs(j))
    return (Ax, rel, b, var_ids)

if __name__ == "__main__":
    graph_filename = "temp_graph.dat"

    N = 10
    UB_ratios = []
    time_ratios = []
    n_vars = 3
    n_cons = 3
    density = 0.8
    feasType = "feas"

    print(f"Going to do {N} trials \n")
    for i in range(N):
        #generate random graph and save to temp_graph.dat
        opb_str = generate_instance(n_vars, n_cons, density, feasType)

        opb_file = f"tmp_instance_{i}.opb"
        with open(opb_file, "w") as f:
            f.write(opb_str)

        blp_instance = readOpb(opb_file)

        (E, VL, LV, c2e, max_var) = reduce_blp_2_dgp(blp_instance)
        G = (sorted(VL.keys()), E, VL, max_var)
        writeDat(G, graph_filename, opb_file)

        
        
        #generate_feasible_dat_graph(graph_filename, n, m , A)

        # ---- Cheap UB ----
        t0 = time.perf_counter()
        cheap_UB = subprocess.run(
            ["upper_bounds//cheap_minErrDGP1_UB", graph_filename],
            check=True,
            text=True,
            capture_output=True
        )
        t1 = time.perf_counter()
        cheap_time = t1 - t0

        # # ---- Gurobi UB ----
        # t0 = time.perf_counter()
        # Gurobi_UB = subprocess.run(
        #     [sys.executable, "upper_bounds//minErrDGP1.py", "temp_graph.dat"],
        #     check=True,
        #     text=True,
        #     capture_output=True
        # )
        # t1 = time.perf_counter()
        # gurobi_time = t1 - t0

        # # ---- Parse outputs ----
        # UB_lines = Gurobi_UB.stdout.strip().splitlines()
        cheap_UB_value = float(cheap_UB.stdout.strip())
        # Gurobi_UB_value = float(UB_lines[-1].split()[-1])

        # UB_ratios.append(cheap_UB_value / Gurobi_UB_value)
        # time_ratios.append(cheap_time / gurobi_time)

        print(
            f"Trial {i}: "
            f"Cheap UB = {cheap_UB_value:.6f} "
            f"(time = {cheap_time:.3f}s), "
            # f"Gurobi UB = {Gurobi_UB_value:.6f} "
            # f"(time = {gurobi_time:.3f}s)\n"
        )
        # if Gurobi_UB_value == 0.0:
        #     print(f"Gurobi UB is zero {Gurobi_UB_value} for trial {i}, cheap UB value {cheap_UB_value}.\n")
        # else:
        #     print(f"(UB ratio = {cheap_UB_value / Gurobi_UB_value:.4f})\n")
        # print(f"(time ratio = {cheap_time / gurobi_time:.4f})\n")

    # avg_UB_ratio = sum(UB_ratios) / N
    # avg_time_ratio = sum(time_ratios) / N
    # print(f"Average UB ratio (Cheap / Gurobi) over {N} trials: {avg_UB_ratio:.4f}")
