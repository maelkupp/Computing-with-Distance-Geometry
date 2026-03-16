import subprocess
import os
import time
import numpy as np
import random
from reductions.blp2dgp_opt import reduce_blp_2_dgp_opt, readOpb, contract_zero_edges
from reductions.blp2dgp import reduce_blp_2_dgp

def writeDat(G, dgpf, opbf):
    (V,E, VL, nlits) = G
    n = max(V)
    with open(dgpf, "w") as f:
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

def create_k_clique_blp(n, p, k):
    """
    n : number of vertices
    p : edge probability
    k : clique size
    """

    edges = set()

    # generate random graph
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if random.random() < p:
                edges.add((i, j))

    lines = []
    # clique size constraint
    sum_line = " ".join(f"1*x{i}" for i in range(1, n+1))
    lines.append(f"{sum_line} = {k};")

    # constraints for non-edges
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if (i, j) not in edges:
                lines.append(f"1*x{i} 1*x{j} <= 1;")

    return "\n".join(lines)

if __name__ == "__main__":
    n = 7
    p = 0.6
    k = 4

    dgpf = "k_clique.dat"
    opbf = "k_clique.opb"

    # blp_str = create_k_clique_blp(n, p, k)
    # with open(opbf, "w") as f:
    #     f.write(blp_str)

    blp_instance = readOpb(opbf)
    (E, VL, LV, c2e, max_var) = reduce_blp_2_dgp_opt(blp_instance)
    (E, VL, LV) = contract_zero_edges(E, VL, LV)

    ## make a graph G=(V,E)
    E = sorted(list(E))
    V = [vid for vid in VL]
    V.sort()
    G = (V,E,VL,max_var)

    ## write DGP1 instance
    writeDat(G, dgpf, opbf)

    solution = subprocess.run(["./solver/solver", dgpf, "0"], capture_output=True, text=True)
    directed_solution = solution.stdout
    print(f"error logs {solution.stderr}")


    print(f"Directed solution:{directed_solution}")

