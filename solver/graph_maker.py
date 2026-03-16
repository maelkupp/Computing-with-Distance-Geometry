import random
import sys



def generate_feasible_dat_graph(filename, n=25, m=38, A=35):
    assert m >= n - 1, "Need at least n-1 edges."

    # Random 1D embedding
    points = (random.sample(range(A), n))

    edges = set()

    # --- Random spanning tree ---
    vertices = list(range(1, n + 1))
    random.shuffle(vertices)

    for i in range(1, n):
        u = vertices[i]
        v = random.choice(vertices[:i])
        edges.add(tuple(sorted((u, v))))

    # --- Add extra edges ---
    while len(edges) < m:
        u, v = random.sample(range(1, n + 1), 2)
        edges.add(tuple(sorted((u, v))))

    # --- Write file ---
    with open(filename, "w") as f:
        f.write("# Feasible DGP instance\n\n")
        f.write("param : E : c I :=\n")

        for (u, v) in edges:
            w = abs(points[u - 1] - points[v - 1])
            f.write(f"  {u} {v}  {w:.3f} 0\n")

        f.write(";\n")


def generate_infeasible_dat_graph(
    filename,
    n=15,
    m=25,
    weight_min=1,
    weight_max=5,
):
    assert m >= n - 1, "Need at least n-1 edges."

    edges = set()

    # --- Random spanning tree ---
    vertices = list(range(1, n + 1))
    random.shuffle(vertices)

    for i in range(1, n):
        u = vertices[i]
        v = random.choice(vertices[:i])
        edges.add(tuple(sorted((u, v))))

    # --- Add extra edges ---
    while len(edges) < m:
        u, v = random.sample(range(1, n + 1), 2)
        edges.add(tuple(sorted((u, v))))

    # --- Write file ---
    with open(filename, "w") as f:
        f.write("# Random (likely infeasible) DGP instance\n\n")
        f.write("param : E : c I :=\n")

        for (u, v) in edges:
            w = random.randint(weight_min, weight_max)
            f.write(f"  {u} {v}  {w:.3f} 0\n")

        f.write(";\n")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python graph_maker.py <filename> <0/1>")
        sys.exit(1)
    filename = sys.argv[1]
    feasible = sys.argv[2] == "1"
    print(f"feasible: {feasible}")
    if feasible:
        generate_feasible_dat_graph(filename)
    else:
        generate_infeasible_dat_graph(filename)
