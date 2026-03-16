import sys
import subprocess
import random
import re
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
from scipy.optimize import minimize



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

def smooth_min_err_dgp1_gurobi(edges, n, epsilon, time_limit=240, mip_gap=0.01):
    #solves the minErrDGP1 with a smooth minimizer instead of the absolute value
    m = gp.Model("SmoothMinErrDGP1")
    m.Params.OutputFlag = 0

    m.Params.TimeLimit = time_limit
    m.Params.MIPGap    = mip_gap
    m.params.Heuristics = 0.7
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = 10

    x = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    #fix the first coordinate vertex at 0 to remove translation symmetry
    m.addConstr(x[0] == 0)
    m.addConstr(x[1] >= 0) #fix the second vertex to be non-negative to remove reflection symmetrys
    err = {}
    for idx, (i, j, d) in enumerate(edges):
        diff = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(diff == x[i] - x[j])

        smooth_absdiff = m.addVar(lb=0)
        m.addQConstr(smooth_absdiff * smooth_absdiff == diff * diff + epsilon * epsilon)

        res = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(res == smooth_absdiff - d)

        err[idx] = m.addVar(lb=0)
        m.addGenConstrAbs(err[idx], res)

    m.setObjective(gp.quicksum(err[idx] for idx in err), GRB.MINIMIZE)
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        embedding = [x[i].X for i in range(n)]
        total_err = sum(abs(abs(embedding[i] - embedding[j]) - d)
                        for i, j, d in edges)
        return total_err, embedding
    else:
        return None, None

def super_smooth_min_err_dgp1_gurobi(edges, n, epsilon, time_limit=240, mip_gap=0.01):
    #solves the minErrDGP1 with a smooth minimizer instead of the absolute value
    m = gp.Model("SmoothMinErrDGP1")
    m.Params.OutputFlag = 0

    m.Params.TimeLimit = time_limit
    m.Params.MIPGap    = mip_gap
    m.params.Heuristics = 0.7
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = 10

    x = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    #fix the first coordinate vertex at 0 to remove translation symmetry
    m.addConstr(x[0] == 0)
    m.addConstr(x[1] >= 0) #fix the second vertex to be non-negative to remove reflection symmetrys
    err = {}
    for idx, (i, j, d) in enumerate(edges):
        diff = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(diff == x[i] - x[j])

        smooth_absdiff = m.addVar(lb=0)
        m.addQConstr(smooth_absdiff * smooth_absdiff == diff * diff + epsilon * epsilon)

        res = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(res == smooth_absdiff - d)

        err[idx] = m.addVar(lb=0)
        m.addGenConstrAbs(err[idx], res)

    m.setObjective(gp.quicksum(err[idx]**2 for idx in err), GRB.MINIMIZE) # add a smooth objective function
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        embedding = [x[i].X for i in range(n)]
        total_err = sum(abs(abs(embedding[i] - embedding[j]) - d)
                        for i, j, d in edges)
        return total_err, embedding
    else:
        return None, None


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

def generate_dat_graph(filename, n=8, m=12, weight_min=1, weight_max=5):
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
            #here attribute a random weight to the edge between weight_min and weight_max
            w = random.randint(weight_min, weight_max)
            f.write(f"  {i} {j}  {w:.3f} 0\n")

        f.write(";\n")


def create_adjacency_list_from_edges(edges, n):
    adj_list = {i: [] for i in range(0, n)}
    for (i, j, w) in edges:
        adj_list[i].append((j, w)) 
        adj_list[j].append((i, w)) 

    return adj_list

def dfs_ordering(adj_list, root):
    visited = set()
    order = []

    stk = [root]
    while stk:
        node = stk.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for (nbr, w) in adj_list[node]:
                if nbr not in visited:
                    stk.append(nbr)
    return order

def feasibility_from_ordering(edges, minim_order):
    adj_list = create_adjacency_list_from_edges(edges, len(minim_order))
    #this is the dictionary that will store the position of each vertex on the line
    pos_map = {minim_order[0]: 0.0} #we can place the first vertex in the minimizing order at position 0 on the line
    index_map = {v: i for i, v in enumerate(minim_order)} #map from vertex id to its index in the minimizing order
    dfs_order = dfs_ordering(adj_list, minim_order[0]) #get a DFS ordering of the vertices starting from the first vertex in the minimizing order
    for i in range(1, len(dfs_order)):
        v = dfs_order[i]
        ngbrs = [nbr for nbr in adj_list[v] if nbr[0] in pos_map] #get the neighbors of v that have already been placed on the line
        if not ngbrs:
            #if v has no neighbors that have been placed on the line yet, we can place it anywhere, we choose to place it at position 0
            pos_map[v] = 0.0
        else:
            if index_map[ngbrs[0][0]] < index_map[v]:
                cand_pos = pos_map[ngbrs[0][0]] + ngbrs[0][1]
            else:
                cand_pos = pos_map[ngbrs[0][0]] - ngbrs[0][1] 
            for j in range(1, len(ngbrs)):
                (nbr, w) = ngbrs[j]
                new_cand_pos = None
                if index_map[nbr] < index_map[v]:
                    new_cand_pos = pos_map[nbr] + w
                else:
                    new_cand_pos = pos_map[nbr] - w
                if abs(new_cand_pos - cand_pos) > 1e-6:
                    #if the new candidate position is not consistent with the previous candidate position, then the ordering is not feasible
                    return False
            pos_map[v] = cand_pos
    return True

    #now that everything has been placed according to the minimizing order we just need to verify that all the distance constraints are satisfied
    return True

def slab_objective(z, edges, n, eps):
    x = z[:n]
    y = z[n:]

    err = 0.0
    for i,j,d in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]

        dist = np.sqrt(dx*dx + dy*dy)
        err += abs(dist - d)

    return err


def solve_slab_local(edges, n, epsilon, eps=1e-8):

    # initial guess
    x0 = np.random.randn(n)
    y0 = np.random.rand(n) * epsilon

    z0 = np.concatenate([x0, y0])

    bounds = [(None,None)]*n + [(0,epsilon)]*n

    res = minimize(
        slab_objective,
        z0,
        args=(edges,n,eps),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter":2000}
    )

    z = res.x
    x = z[:n]
    y = z[n:]
    emb = [(x[i], y[i]) for i in range(len(x))]


    return res.fun, emb



if __name__ == "__main__":
    #create a random DGP instance and obtain its edges and vertices
    if len(sys.argv) != 2:
        print("Usage: python -m orderings.obtain_minimising_ordering infeas/feas")
        sys.exit(1)
    
    if sys.argv[1] != "feas" and sys.argv[1] != "infeas":
        print("Usage: python -m orderings.obtain_minimising_ordering infeas/feas")
        sys.exit(1)
    
    if sys.argv[1] == "feas":
        generate_feasible_dat_graph("temp_graph.dat", n=8, m=14, A=25)
    else:
        #not necessarily an infeasible instance but highly likely to be one
        generate_dat_graph("temp_graph.dat", n=5, m=8, weight_min=1, weight_max=10)  

    edges, n, vertex_names = parse_dgp_dat_file("temp_graph.dat")

    #solve the DGP instance with Gurobi, to obtain an upper bound and the embedding of the vertices on the line
    t0 = time.time()
    #ub, emb = min_err_dgp1_gurobi(edges, n, time_limit=60, mip_gap=0.05)
    t1 = time.time()
    print(f"Time taken to solve minErrDGP1: {t1 - t0:.2f} seconds", file=sys.stderr)
    #print(f"Upper bound objective = {ub:.6f}", file=sys.stderr)
    #print(f"emd: {emb} with error {ub}", file=sys.stderr) #sort the embedding by the x coordinate of the vertices

    # minimizing_order = sorted(range(n), key=lambda i: emb[i])
    # print(f"minimizing order: {minimizing_order}", file=sys.stderr)


    epsilons = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001,]

    for epsilon in epsilons:
        t0 = time.time()
        ub, emb = smooth_min_err_dgp1_gurobi(edges, n, epsilon)
        t1 = time.time()
        print(f"Time taken to solve smooth minErrDGP1 with epsilon {epsilon}): {t1 - t0:.2f} seconds", file=sys.stderr)
        print(f"Upper bound objective with slab of height {epsilon} = {ub:.6f}", file=sys.stderr)
        print(f"emd with slab of height {epsilon}: with error {ub}", file=sys.stderr)
        minim_order = sorted(range(n), key=lambda i: emb[i]) #sort wrt to the x coordinate, we do a projection

        is_feas = feasibility_from_ordering(edges, minim_order)
        print(f"Is the minimizing order from epsilon {epsilon} feasible? {is_feas} \n \n", file=sys.stderr)





