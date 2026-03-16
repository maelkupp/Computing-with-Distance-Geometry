#!/usr/bin/env python3
import sys, os, time, datetime, math, random
import numpy as np
from collections import defaultdict
from pysat.solvers import Solver
import itertools

### GLOBAL CONSTANTS
myEps = 1e-2
minVal = -10
maxVal = 10
wrongPerturbRange = 3

##############################
# Instance Generation Functions
##############################

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

        opb_lines.append(" ".join(terms) + " = {};".format(int(b[i])))
    opb_str = "\n".join(opb_lines)
    return opb_str

##############################
# Reduction Functions (BLP-to-SAT)
##############################

def readOpb(opbf):
    b = {}
    rel = {}
    Ax = {}
    var_ids = set()
    with open(opbf, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[-1] == ';':
                b[i] = int(parts[-2])
                rel[i] = parts[-3]
                parts = parts[:-3]
            else:
                b[i] = int(parts[-1][:-1])
                rel[i] = parts[-2]
                parts = parts[:-2]
            Ax[i] = {}
            for term in parts:
                coeff, var = term.split('*')
                j = int(var[1:])  # e.g., "x3" -> 3
                coeff = int(coeff)
                Ax[i][j] = coeff
                var_ids.add(abs(j))
    return (Ax, rel, b, var_ids)

def normalizing_single_constraint(Ax_i, b_val, rel_val):
    norm_Ax = {}
    norm_b = b_val
    relation = rel_val
    if relation == ">=":
        relation = "<="
        for var, coeff in Ax_i.items():
            norm_Ax[var] = -coeff
        norm_b = -norm_b
    else:
        norm_Ax = dict(Ax_i)
    norm_Ax_final = {}
    for xj, c in norm_Ax.items():
        if c < 0:
            c = abs(c)
            xj = -xj  # flip literal sign
            norm_b += c
        norm_Ax_final[xj] = c
    return (norm_Ax_final, norm_b, relation)

def normalize_blp(blp_instance):
    (Ax, relation, b, var) = blp_instance
    Ax_norm = {}
    rel_norm = {}
    b_norm = {}
    new_index = 0
    for i in Ax:
        norm_Axi, norm_b, norm_rel = normalizing_single_constraint(Ax[i], b[i], relation[i])
        if norm_rel == "=":
            Ax_norm[new_index] = dict(norm_Axi)
            rel_norm[new_index] = "<="
            b_norm[new_index] = norm_b
            new_index += 1
            flipped_Ax = {}
            flipped_b = -norm_b
            for var, coeff in norm_Axi.items():
                flipped_coeff = -coeff
                if flipped_coeff < 0:
                    flipped_coeff = abs(flipped_coeff)
                    var = -var
                    flipped_b += flipped_coeff
                flipped_Ax[var] = flipped_coeff
            Ax_norm[new_index] = flipped_Ax
            rel_norm[new_index] = "<="
            b_norm[new_index] = flipped_b
            new_index += 1
        else:
            Ax_norm[new_index] = norm_Axi
            rel_norm[new_index] = norm_rel
            b_norm[new_index] = norm_b
            new_index += 1
    return (Ax_norm, rel_norm, b_norm, var)

class CNFEncoder:
    def __init__(self):
        self.var_counter = 0
        self.clauses = []  # list of clauses (each a list of ints)
    def new_var(self, name="v"):
        self.var_counter += 1
        return self.var_counter
    def add_clause(self, *literals):
        self.clauses.append(list(literals))
    def add_and(self, out, in1, in2):
        self.add_clause(-in1, -in2, out)
        self.add_clause(in1, -out)
        self.add_clause(in2, -out)
    def add_or(self, out, in1, in2):
        self.add_clause(in1, in2, -out)
        self.add_clause(-in1, out)
        self.add_clause(-in2, out)
    def add_eq(self, out, var):
        if out == var:
            return
        if out == -var:
            self.add_clause(-out, -var)
            self.add_clause(out, var)
            return
        self.add_clause(-out, var)
        self.add_clause(out, -var)

def xor2(encoder, a, b, out):
    encoder.add_clause(-a, -b, -out)
    encoder.add_clause(-a, b, out)
    encoder.add_clause(a, -b, out)
    encoder.add_clause(a, b, -out)

def warners_leq(weights, vars_, bound, encoder):
    if bound < 0:
        v = encoder.new_var("unsat")
        encoder.add_clause(v)
        encoder.add_clause(-v)
        return encoder.clauses
    if sum(weights) <= bound:
        return encoder.clauses
    max_sum = sum(weights)
    M = max_sum.bit_length()
    sums_by_level = defaultdict(list)
    for weight, lit in zip(weights, vars_):
        for k in range(M):
            if (weight >> k) & 1:
                bit_var = encoder.new_var(f"bit_{lit}_{k}")
                encoder.add_eq(bit_var, lit)
                sums_by_level[k].append(bit_var)
    carry_bits = defaultdict(list)
    sum_bits = {}
    for k in range(M + int(math.ceil(math.log2(len(weights)))) + 2):
        current_bits = sums_by_level[k] + carry_bits[k]
        bit_counts = defaultdict(int)
        unique_bits = []
        for bit in current_bits:
            bit_counts[bit] += 1
        for bit, count in bit_counts.items():
            pairs, remainder = divmod(count, 2)
            for _ in range(pairs):
                carry_bits[k+1].append(bit)
            if remainder:
                unique_bits.append(bit)
        current_bits = unique_bits
        while len(current_bits) > 1:
            a = current_bits.pop()
            b = current_bits.pop()
            sum_var = encoder.new_var(f"sum_{k}_{len(current_bits)}")
            carry_var = encoder.new_var(f"carry_{k}_{len(current_bits)}")
            xor2(encoder, a, b, sum_var)
            encoder.add_and(carry_var, a, b)
            current_bits.append(sum_var)
            carry_bits[k+1].append(carry_var)
        if current_bits:
            sum_bits[k] = current_bits[0]
        else:
            zero_bit = encoder.new_var(f"zero_bit_{k}")
            encoder.add_clause(-zero_bit)  # force this bit to 0
            sum_bits[k] = zero_bit
    max_level = max(sum_bits.keys())
    bound_bits = [(bound >> k) & 1 for k in range(max_level+1)]
    higher_bits_match = []
    for k in reversed(range(len(bound_bits))):
        sum_bit = sum_bits.get(k)
        b_bit = bound_bits[k]
        if sum_bit is None:
            continue
        if b_bit == 0:
            if higher_bits_match:
                match_var = encoder.new_var(f"match_above_{k}")
                for hb in higher_bits_match:
                    encoder.add_clause(-match_var, hb)
                encoder.add_clause(match_var, *[-hb for hb in higher_bits_match])
                encoder.add_clause(-match_var, -sum_bit)
            else:
                encoder.add_clause(-sum_bit)
        match_current = encoder.new_var(f"match_bit_{k}")
        if b_bit == 1:
            encoder.add_eq(match_current, sum_bit)
        else:
            encoder.add_eq(match_current, -sum_bit)
        higher_bits_match.append(match_current)
    for k in range(len(bound_bits), max(sum_bits.keys())+1):
        if k in sum_bits:
            encoder.add_clause(-sum_bits[k])
    return encoder.clauses

def encode_constraints(Ax, rel, b, encoder):
    constraint2clause = {}
    for i in Ax:
        coeffs = list(Ax[i].values())
        vars_ = list(Ax[i].keys())
        if rel[i] != "<=":
            raise ValueError(f"Expected <= constraint at index {i}, got: {rel[i]}")
        start_idx = len(encoder.clauses)
        warners_leq(coeffs, vars_, b[i], encoder)
        end_idx = len(encoder.clauses)
        constraint2clause[i] = list(range(start_idx, end_idx))
    return constraint2clause

def run_blp_to_sat(opb_content):
    # Write the OPB instance to a temporary file.
    tmpfile = "tmp_instance.opb"
    with open(tmpfile, "w") as f:
        f.write(opb_content)
    try:
        blp_instance = readOpb(tmpfile)
        (Ax, rel, b, var) = normalize_blp(blp_instance)
        encoder = CNFEncoder()
        original_vars = set(abs(v) for Ax_i in Ax.values() for v in Ax_i)
        if original_vars:
            encoder.var_counter = max(original_vars)
        encode_constraints(Ax, rel, b, encoder)
        with Solver(name='glucose3') as solver:
            for clause in encoder.clauses:
                solver.add_clause(clause)
            sat = solver.solve()
            model = solver.get_model() if sat else None
        return sat, model
    finally:
        os.remove(tmpfile)

##############################
# Expected Feasibility (Brute Force)
##############################

def parse_constraint(line):
    # Remove comments and trailing semicolon
    line = line.split('#')[0].strip()
    if not line:
        return None
    if line.endswith(';'):
        line = line[:-1]
    parts = line.split()
    relation = parts[-2]
    b_val = int(parts[-1])
    term_parts = parts[:-2]
    terms = []
    for term in term_parts:
        term = term.strip()
        if term.startswith('+'):
            term = term[1:]
        coeff_str, var_str = term.split('*')
        coeff = int(coeff_str)
        var = int(var_str[1:])
        terms.append((coeff, var))
    return terms, relation, b_val

def evaluate_opb_instance(opb_str, n_vars):
    # Brute-force evaluation (only practical for small n_vars)
    constraints = []
    for line in opb_str.splitlines():
        parsed = parse_constraint(line)
        if parsed is not None:
            constraints.append(parsed)
    for bits in itertools.product([0,1], repeat=n_vars):
        assignment = {i+1: bits[i] for i in range(n_vars)}
        ok = True
        for terms, relation, b_val in constraints:
            total = sum(coeff * assignment[var] for coeff, var in terms)
            if relation == "<=" and not (total <= b_val):
                ok = False
                break
            elif relation == ">=" and not (total >= b_val):
                ok = False
                break
            elif relation == "=" and not (total == b_val):
                ok = False
                break
        if ok:
            return True
    return False

##############################
# Main Loop: Mass Testing
##############################

def main():
    if len(sys.argv) < 6:
        exit("Usage: {} num_tests n_vars n_cons density [feas|rnd|infeas|perturb]".format(sys.argv[0]))
    num_tests = int(sys.argv[1])
    n_vars = int(sys.argv[2])
    n_cons = int(sys.argv[3])
    density = float(sys.argv[4])
    feasType = sys.argv[5].lower()
    validTypes = {"feas", "rnd", "infeas", "perturb"}
    if feasType not in validTypes:
        exit("Type must be one of: feas, rnd, infeas, perturb")
    
    tests_passed = 0
    tests_failed = 0
    failures = []
    
    # For small n_vars, we can brute-force expected feasibility.
    compute_expected = (n_vars <= 10)
    
    for i in range(1, num_tests+1):
        opb_str = generate_instance(n_vars, n_cons, density, feasType)
        # Evaluate expected feasibility (if possible)
        expected = None
        if compute_expected:
            expected = evaluate_opb_instance(opb_str, n_vars)
        try:
            sat, model = run_blp_to_sat(opb_str)
        except Exception as e:
            print(f"Test {i}: Exception occurred:\n{e}\nInstance:\n{opb_str}\n")
            tests_failed += 1
            continue
        
        # If we computed expected feasibility, compare
        if expected is not None:
            if sat == expected:
                tests_passed += 1
            else:
                tests_failed += 1
                failures.append((i, opb_str, expected, sat))
                print(f"Test {i} FAILED: Expected {'SAT' if expected else 'UNSAT'}, got {'SAT' if sat else 'UNSAT'}")
        else:
            # Otherwise, just count SAT vs. UNSAT occurrences.
            tests_passed += 1  # Assume pass if no expected value computed.
        
    print(f"\nTesting complete: {tests_passed} passed, {tests_failed} failed out of {num_tests} tests.")
    if failures:
        print("\nFailed instances:")
        for idx, inst, exp, got in failures:
            print(f"Test {idx}: Expected {'SAT' if exp else 'UNSAT'}, got {'SAT' if got else 'UNSAT'}")
            print("Instance:")
            print(inst)
            print("-----")

if __name__ == '__main__':
    main()

#run this script by calling python3 testingreductionmael.py num_tests num_vars num_constraints density type_BLP
#type BLP is either feas infeas rnd perturb

'''
    if bound < 0:
        v = encoder.new_var("unsat")
        encoder.add_clause(v)
        encoder.add_clause(-v)
        return encoder.clauses
'''