import re
import gurobipy as gp
from gurobipy import GRB
import sys
import math
import random
import time


def parse_dgp_dat_file(dat_file):
    edges = []
    n = 0
    in_edges = False

    vertex_names = {}  # id (0-based) -> name

    with open(dat_file, "r") as f:
        for line in f:
            if line.strip().startswith("param : E : c I :="):
                in_edges = True
                continue

            if in_edges:
                if line.strip().startswith(";"):
                    break

                # match edge data
                m = re.match(r"\s*(\d+)\s+(\d+)\s+([0-9eE\+\-\.]+)", line)
                if not m:
                    continue

                i, j, dist = int(m.group(1)), int(m.group(2)), float(m.group(3))
                i0, j0 = i - 1, j - 1
                edges.append((i0, j0, dist))
                n = max(n, i, j)

                # extract names after # [name_i,name_j]
                name_match = re.search(r"#\s*\[([^\]]+)\]", line)
                if name_match:
                    names = name_match.group(1).split(",")
                    if len(names) == 2:
                        name_i, name_j = names[0].strip(), names[1].strip()
                        vertex_names.setdefault(i0, name_i)
                        vertex_names.setdefault(j0, name_j)

    return edges, n, vertex_names

def min_err_dgp1_slab_gurobi(edges, n, epsilon, time_limit=240, mip_gap=0.01):
    #solves the minErrDGP1 problem with a slab of height epsilon, solving a 2D instance on a restricted domain of the plane of height epsilon
    m = gp.Model("MinErrDGP1")
    m.Params.OutputFlag = 0

    m.Params.TimeLimit = time_limit
    m.Params.MIPGap    = mip_gap
    m.params.Heuristics = 0.7
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = 10

    x = m.addVars(n, 2, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    #fix the first coordinate vertex at 0 to remove translation symmetry
    for i in range(n):
        m.addConstr(x[i, 1] <= epsilon) #restrict to a slab of height epsilon
        m.addConstr(x[i, 1] >= 0) #restrict to a slab of height epsilon


    m.addConstr(x[0, 0] == 0)
    m.addConstr(x[0, 1] == 0) #also fix the second coordinate of the first vertex to remove some symmetries
    m.addConstr(x[1, 0] >= 0) #fix the second vertex to be non-negative in the second coordinate to remove reflection symmetry across the x-axis
    m.addConstr(x[1, 1] >= 0) #fix the second vertex to be non-negative in the second coordinate to remove reflection symmetry across the x-axis

    err = {}
    for idx, (i, j, d) in enumerate(edges):
        diff_x = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        diff_y = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(diff_x == x[i, 0] - x[j, 0])
        m.addConstr(diff_y == x[i, 1] - x[j, 1])

        absdiff_x = m.addVar(lb=0)
        absdiff_y = m.addVar(lb=0)
        m.addGenConstrAbs(absdiff_x, diff_x)
        m.addGenConstrAbs(absdiff_y, diff_y)

        aux = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(aux == (x[i,0] - x[j,0])*(x[i,0] - x[j,0])+(x[i,1] - x[j,1])*(x[i,1] - x[j,1]) - d*d)

        err[idx] = m.addVar(lb=0)
        m.addGenConstrAbs(err[idx], aux)

    m.setObjective(gp.quicksum(err[idx] for idx in err), GRB.MINIMIZE)
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        embedding = [(x[i, 0].X, x[i, 1].X) for i in range(n)]
        total_err = sum(abs(((embedding[i][0] - embedding[j][0]) ** 2 + (embedding[i][1] - embedding[j][1]) ** 2) ** 0.5 - d)
                        for i, j, d in edges)
        return total_err, embedding
    else:
        return None, None
    

def min_err_dgp1_gurobi(edges, n, time_limit=240, mip_gap=0.01):
    m = gp.Model("MinErrDGP1")
    m.Params.OutputFlag = 0

    m.Params.TimeLimit = time_limit
    m.Params.MIPGap    = mip_gap
    m.Params.Heuristics = 0.7
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = 10

    x = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    
    #add the following constraints to speed up processing by removing symmetries
    m.addConstr(x[0] == 0) #fix the first vertex at 0 to remove translation symmetry
    m.addConstr(x[1] >= 0) #fix the second vertex to be non-negative to remove reflection symmetrys


    err = {}
    for idx, (i, j, d) in enumerate(edges):
        print(f"Looking at edge idx {idx} between vertex {i} and vertex {j} with distance {d}", file=sys.stderr)
        diff = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(diff == x[i] - x[j])

        absdiff = m.addVar(lb=0)
        m.addGenConstrAbs(absdiff, diff)

        aux = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(aux == absdiff - d)

        err[idx] = m.addVar(lb=0)
        m.addGenConstrAbs(err[idx], aux)

    m.setObjective(gp.quicksum(err[idx] for idx in err), GRB.MINIMIZE)
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        embedding = [x[i].X for i in range(n)]
        total_err = sum(abs(abs(embedding[i] - embedding[j]) - d)
                        for i, j, d in edges)
        return total_err, embedding
    else:
        return None, None


def generate_random_cycle_dat_file(filename, n=5, weight_min=1, weight_max=5):
    with open(filename, "w") as f:
        f.write("# DGP instance in .dat format for dgp.mod\n")
        f.write("# this graph is a cycle on n vertices with random edge weights\n\n")

        f.write("param : E : c I :=\n")
        for i in range(n):
            j = (i + 1) % n
            w = random.randint(weight_min, weight_max)
            f.write(f"  {i+1} {j+1}  {w:.3f} 0\n")
        f.write(";\n")
if __name__ == "__main__":

    edges, n, vertex_names = parse_dgp_dat_file(sys.argv[1])
    t0 = time.time()
    ub, emb = min_err_dgp1_gurobi(edges, n, time_limit=60, mip_gap=0.05)
    t1 = time.time()

    print(f"Upper bound objective = {ub:.6f}", file=sys.stderr)
    for i, xi in enumerate(emb):
        name = vertex_names.get(i, f"v{i}")
        print(f"  {name:12s} (id {i}) = {xi:.6f}", file=sys.stderr)

    #return just the upper bound value to the orchestrating script
    print(f"{ub:.6f} {t1-t0:.2f}")