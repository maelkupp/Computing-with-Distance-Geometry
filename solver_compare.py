import subprocess
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

from reductions.blp2dgp import reduce_blp_2_dgp
from reductions.blp2dgp_opt import reduce_blp_2_dgp_opt, contract_zero_edges


### GLOBAL CONSTANTS
myEps = 1e-2
minVal = -10
maxVal = 10
wrongPerturbRange = 3

##################

def run_dgp_solver(dat_file):


    result = subprocess.run(
        ["./solver/solver", dat_file, "0"],
        capture_output=True,
        text=True
    )


    output = result.stdout
    print(f"Solver output: {output}")
    print(f"logs: {result.stderr}")

    try:
        value = float(output.split()[-2])
        runtime = float(output.split()[-1])
    except:
        value = None
        runtime = None

    return value, runtime

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


def solve_minerr_blp_gurobi(opb_file):
    Ax, rel, b, var = readOpb(opb_file)

    start = time.time()

    model = gp.Model("MinErrBLP")
    model.Params.OutputFlag = 0

    # binary variables
    x = {j: model.addVar(vtype=GRB.BINARY, name=f"x{j}") for j in var}

    # error variables
    e = {i: model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"e{i}") for i in Ax}

    model.update()

    for i in Ax:
        expr = gp.LinExpr()

        for lit, coeff in Ax[i].items():
            j = abs(lit)

            if lit > 0:
                expr += coeff * x[j]
            else:
                expr += coeff * (1 - x[j])

        if rel[i] == "<=":
            model.addConstr(expr <= b[i] + e[i])

        elif rel[i] == ">=":
            model.addConstr(expr >= b[i] - e[i])

        elif rel[i] == "=":
            model.addConstr(expr <= b[i] + e[i])
            model.addConstr(expr >= b[i] - e[i])

    model.setObjective(gp.quicksum(e[i] for i in e), GRB.MINIMIZE)

    model.optimize()

    runtime = time.time() - start

    return model.objVal, runtime

def solve_dgp_instance(dat_file, solver_path="upper_bounds/solver"):
    
    start = time.time()

    result = subprocess.run(
        [solver_path, dat_file],
        capture_output=True,
        text=True
    )

    runtime = time.time() - start

    output = result.stdout.strip()

    try:
        value = float(output.split()[-1])
    except:
        value = None

    return value, runtime

def solve_via_dgp(opb_file):

    blp_instance = readOpb(opb_file)

    (E, VL, LV, c2e, max_var) = reduce_blp_2_dgp_opt(blp_instance)
    (E, VL, LV) = contract_zero_edges(E, VL, LV)

    E = sorted(list(E))
    V = sorted(VL.keys())
    G = (V, E, VL, max_var)

    dat_file = opb_file.replace(".opb", "-dgp.dat")

    writeDat(G, dat_file, opb_file)

    dgp_val, solver_time = run_dgp_solver(dat_file)


    return dgp_val, solver_time

def main():

    if len(sys.argv) < 6:
        exit("Usage: {} num_tests n_vars n_cons density [feas|rnd|infeas|perturb]".format(sys.argv[0]))
        return

    num_tests = int(sys.argv[1])
    n_vars = int(sys.argv[2])
    n_cons = int(sys.argv[3])
    density = float(sys.argv[4])
    feasType = sys.argv[5].lower()

    validTypes = {"feas", "rnd", "infeas", "perturb"}
    if feasType not in validTypes:
        exit("Type must be one of: feas, rnd, infeas, perturb")

    for i in range(1, num_tests + 1):

        print(f"\nTest {i}")

        opb_str = generate_instance(n_vars, n_cons, density, feasType)

        opb_file = f"tmp_instance_{i}.opb"
        with open(opb_file, "w") as f:
            f.write(opb_str)

        # ------------------
        # Solve BLP
        # ------------------


        # ------------------
        # Solve via DGP
        # ------------------

        blp_instance = readOpb(opb_file)

        (E, VL, LV, c2e, max_var) = reduce_blp_2_dgp_opt(blp_instance)
        (E, VL, LV) = contract_zero_edges(E, VL, LV)
        E = sorted(list(E))
        V = sorted(VL.keys())
        G = (V, E, VL, max_var)
        dat_file = opb_file.replace(".opb", "-dgp.dat")
        writeDat(G, dat_file, opb_file)
        dgp_val, solver_time = run_dgp_solver(dat_file)

        print(f"dgp value {dgp_val}")

        (E, VL, LV, c2e, max_var) = reduce_blp_2_dgp(blp_instance)
        (E, VL, LV) = contract_zero_edges(E, VL, LV)
        E = sorted(list(E))
        V = sorted(VL.keys())
        G = (V, E, VL, max_var)
        dat_file = opb_file.replace(".opb", "-dgp.dat")
        writeDat(G, dat_file, opb_file)
        dgp_val, solver_time = run_dgp_solver(dat_file)


        


if __name__ == "__main__":
    main()
