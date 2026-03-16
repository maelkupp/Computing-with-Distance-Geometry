################ IMPORTS ##################

import sys
import os
import time
import subprocess
import datetime
import mimetypes
import re
import pickle
import math
import numpy as np
from collections import defaultdict
import math

####################### GLOBALS ###############################

myEps = 1e-2
myZero = 1e-9

satHeader = "p cnf"
verbose = 1 
verbose = 0 # quiet

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).isoformat().replace('-','').replace('T','').replace(':','').split('.')[0][2:]
exn = sys.argv[0]

##################### FUNCTIONS ###############################3
## read an OPB file Ax rel b (assume integer coeffs)
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


## write an AMPL .dat file encoding the DGP by list of edges
def writeDat(G, dgpf, opbf, e=exn, t=timestamp):
    (V,E, VL, nlits) = G
    n = max(V)
    with open(dgpf, "w") as f:
        print("# DGP instance in .dat format for dgp.mod", file=f)
        print("# written by {} on {}".format(e, t), file=f)
        print("# this graph is a DGP encoding of the BLP instance", file=f)
        print("#   found in {}".format(opbf), file=f)
        print("param n := {};".format(n), file=f)
        print("param nlits := {};".format(nlits), file=f)
        print("param Kdim := 1;", file=f)
        print("param : E : c I :=", file=f)

        for el in E:
            print("  {} {}  {:.3f} {}  # [{},{}]".format(el[0],el[1],el[2],el[3],VL[el[0]],VL[el[1]]), file=f)
        print(";", file=f)
        #print("# vertex map", file=f)
        #for i in V:
        #    print("# vertex {} is {}".format(i, VL[i]), file=f)
        print(f"Successfully writen DGP instance to {dgpf}")
    return

def normalizing_single_constraint(Ax_i, b, rel):
    """
    Normalizes a single BLP constraint so that:
      - The inequality is in the form "≤".
      - All coefficients are positive.
    If the original relation is "≥", the function multiplies both sides by -1.
    Additionally, if any coefficient is negative, it flips the corresponding literal
    and adds the absolute value of the coefficient to the bound.
    """

    norm_Ax = {}
    norm_b = b
    relation = rel

    if relation == ">=":
        # Multiply both sides by -1
        relation = "<="
        for var, coeff in Ax_i.items():
            norm_Ax[var] = -coeff
        norm_b = -norm_b
    else:
        norm_Ax = dict(Ax_i)  # copy

    #make sure all the coefficients are positive
    norm_Ax_final = {}
    for xj, c in norm_Ax.items():
        if c < 0:
            # Flip variable sign if coeff is negative
            c = abs(c)
            xj = -xj #make the literal a 'not' literal
            
            norm_b += c #add the coefficient over to the other side
        norm_Ax_final[xj] = c
    
    return (norm_Ax_final, norm_b, relation)

def normalize_blp(blp_instance):
    #obtain a blp_instance of form (Ax, relation, b, var)
    #return an equivalent blp_instance where all the coefficients are positive and all the relations are <=
    # the equality constraint is written as <= and a >= constraint
    (Ax, relation, b, var) = blp_instance
    Ax_norm = {}
    rel_norm = {}
    b_norm = {}
    
    new_index = 0  # Use a new index to store normalized constraints

    for i in Ax:
        norm_Axi, norm_b, norm_rel = normalizing_single_constraint(Ax[i], b[i], relation[i])

        # Otherwise, just store the constraint as-is
        Ax_norm[new_index] = norm_Axi
        rel_norm[new_index] = norm_rel
        b_norm[new_index] = norm_b
        new_index += 1
    
    return (Ax_norm, rel_norm, b_norm, var)


def get_literal_coeff(Ax, index, j):
    #function looks at the index-th constraint of the BLP instance Ax and look at the j-th literal, checking for both negated and original literals
    if Ax[index].get(j, 0) != 0:
        #it was a literal
        return Ax[index].get(j, 0)
    elif Ax[index].get(-j, 0) != 0:
        #it was a negated literal
        return Ax[index].get(-j, 0)
    else:
        #we know that the coefficient is equal to 0, but that doesnt matter
        return 0
    
def get_literal_sign(Ax, index, j):
    #function looks at the index-th constraint of the BLP instance Ax and look at the j-th literal sends True if it is a literal and False if it is a negated literal
    #if the coefficient is 0, we do not care and simply return True
    if Ax[index].get(j, 0) > 0:
        return True
    if Ax[index].get(-j, 0) > 0:
        return False
    return True


def multiply_blp_n(Ax, b, n):
    for index in range(len(Ax)):
        b[index] *= n
        keys = []
        for key in Ax[index]:
            keys.append(key)
        for k in keys:
            Ax[index][k] *= n
        print(Ax[index])
        print(b[index])
    return (Ax, b)

def reduce_blp_2_dgp(blp_instance):
    (Ax, rel, b, var) = normalize_blp(blp_instance)

    (Ax, b) = multiply_blp_n(Ax, b, 5)
    max_var = max(var) # here we obtain the number of literals in our BLP instance
    nconstraints = len(Ax) #the number of DGP contraints we have to map to

    c2e = {}
    E = set([])  # weighted edges of the DGP graph
    VL = {} #vertex_id to literal name
    LV = {} #literal name to vertex_id
    vertex_id = 1

    ## A vertices
    VL[vertex_id] = "A"
    LV[VL[vertex_id]] = vertex_id
    vertex_id += 1

    VL[vertex_id] = "P_0"
    LV[VL[vertex_id]] = vertex_id
    vertex_id += 1
    edges_to_add = [(LV["A"], LV["P_0"], 1, 0)] # initiating our edges to add list with the edge from A to the start of the partial sum for this BLP 

    ## literal vertices
    for i in range(1,max_var+1):
        VL[vertex_id] = "X+{}".format(i) #the literal
        LV[VL[vertex_id]] = vertex_id
        vertex_id += 1
        VL[vertex_id] = "X-{}".format(i) #the negation of the literal
        LV[VL[vertex_id]] = vertex_id
        vertex_id += 1


    E.update([(LV["A"], LV["X+{}".format(i)], 1.0, 0) for i in range(1,max_var+1)]) # add the weight from the vertex A to each vertex encoding the literals
    E.update([(LV["A"], LV["X-{}".format(i)], 1.0, 0) for i in range(1,max_var+1)]) # add the weight from the vertex A to each vertex encoding the negation of the literal
    E.update([(LV["X-{}".format(i)], LV["X+{}".format(i)], 2, 0) for i in range(1,max_var+1)]) # add the edge of weight two between the vertex of the literal and the vertex of the negated literal



    for index in Ax:
        if b[index] != 0 or rel[index] == "=":
            VL[vertex_id] = "T^{}".format(index) #the target vector T
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1

        if b[index] > 0:
            edges_to_add.append((LV["A"], LV["T^{}".format(index)], b[index]+1, 0)) #forces the b vertex to fold in the direction of the sum
            edges_to_add.append((LV["P_0"], LV["T^{}".format(index)], b[index], 0)) 
        elif b[index] < 0:
            edges_to_add.append((LV["A"], LV["T^{}".format(index)],  -b[index] - 1, 0)) #forces the T vertex to fold in the opposite direction to the direction of the sum
            edges_to_add.append((LV["P_0"], LV["T^{}".format(index)], -b[index], 0))

    for index in Ax:
        #iterating over each constraint in the BLP

        #################### ENCODING THE PARTIAL SUM and the BOUND #############################


        VL[vertex_id] = "M_1^{}".format(index)
        LV[VL[vertex_id]] = vertex_id
        vertex_id += 1

        edges_to_add.append((LV["A"], LV["M_1^{}".format(index)], (get_literal_coeff(Ax, index, 1)/2) +1, 0))
        VL[vertex_id] = "L_1^{}".format(index)
        LV[VL[vertex_id]] = vertex_id
        vertex_id += 1

        VL[vertex_id] = "A_2^{}".format(index)
        LV[VL[vertex_id]] = vertex_id
        vertex_id += 1


        edges_to_add.append((LV["P_0"], LV["M_1^{}".format(index)], (get_literal_coeff(Ax, index, 1))/2, 0))
        edges_to_add.append((LV["A"], LV["L_1^{}".format(index)], (get_literal_coeff(Ax, index, 1))/2, 0))
        edges_to_add.append((LV["A_2^{}".format(index)], LV["L_1^{}".format(index)], (get_literal_coeff(Ax, index, 1))/2, 0))  

        for i in range(1, max_var):
            VL[vertex_id] = "P_{}^{}".format(i, index)
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1

            VL[vertex_id] = "M_{}^{}".format(i+1, index)
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1

            VL[vertex_id] = "L_{}^{}".format(i+1, index)
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1

            VL[vertex_id] = "A_{}^{}".format(i+2, index)
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1

            edges_to_add.append((LV["M_{}^{}".format(i, index)], LV["P_{}^{}".format(i,index)], get_literal_coeff(Ax, index, i)/2, 0)) #link between the last M_i and the partial sum P_i
            edges_to_add.append((LV["P_{}^{}".format(i, index)], LV["M_{}^{}".format(i+1,index)], get_literal_coeff(Ax, index, i+1)/2, 0)) #link between this P_i and M_i+1
            edges_to_add.append((LV["A_{}^{}".format(i+1, index)], LV["P_{}^{}".format(i,index)], 1, 0))
            edges_to_add.append((LV["A_{}^{}".format(i+1, index)], LV["M_{}^{}".format(i+1,index)], get_literal_coeff(Ax, index, i+1)/2 + 1, 0))
            edges_to_add.append((LV["A_{}^{}".format(i+1, index)], LV["L_{}^{}".format(i+1,index)], get_literal_coeff(Ax, index, i+1)/2, 0))
            edges_to_add.append((LV["L_{}^{}".format(i+1, index)], LV["A_{}^{}".format(i+2,index)], get_literal_coeff(Ax, index, i+1)/2, 0))

        VL[vertex_id] = "P_{}^{}".format(max_var, index)
        LV[VL[vertex_id]] = vertex_id
        vertex_id += 1
        edges_to_add.append((LV["M_{}^{}".format(max_var, index)], LV["P_{}^{}".format(max_var,index)], get_literal_coeff(Ax, index, max_var)/2, 0))
        edges_to_add.append((LV["A_{}^{}".format(max_var+1, index)], LV["P_{}^{}".format(max_var,index)], 1, 0))
        #here we have created the system of partial sums we now need to encode the buffer

        ############ ENCODING THE BUFFER ##########################################
        if b[index] > 0 and rel[index] == "<=":
            l = math.floor(math.log2(b[index]))+1 #obtain the floor of log base 2  + 1 of our bound to encode the buffer
            VL[vertex_id] = "R_{}^{}".format(0, index)
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1

            edges_to_add.append((LV["P_{}^{}".format(max_var, index)], LV["R_{}^{}".format(0,index)], 1/2, 0))

            for i in range(0, l):
                if i == 1:
                    #will just hard code exactly what I need to add, otherwise will get too complicated
                    VL[vertex_id] = "R_.5^{}".format(index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1 

                    VL[vertex_id] = "S_.5^{}".format(index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    VL[vertex_id] = "S_1^{}".format(index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    VL[vertex_id] = "R_2^{}".format(index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    edges_to_add.append((LV["S_0^{}".format(index)], LV["R_.5^{}".format(index)], 0.5 ,0))  
                    edges_to_add.append((LV["R_.5^{}".format(index)], LV["S_.5^{}".format(index)], 0.5 ,0))
                    edges_to_add.append((LV["S_.5^{}".format(index)], LV["R_1^{}".format(index)], 0.5 ,0))
                    edges_to_add.append((LV["R_1^{}".format(index)], LV["S_1^{}".format(index)], 0.5 ,0))
                    edges_to_add.append((LV["S_1^{}".format(index)], LV["R_2^{}".format(index)], 2,0)) #first encoding the upper line (the partial sum)

                    edges_to_add.append((LV["A_{}^{}".format(max_var + 1.5, index)], LV["S_0^{}".format(index)], 1, 0))
                    edges_to_add.append((LV["A_{}^{}".format(max_var + 1.5, index)], LV["R_.5^{}".format(index)], 1.5, 0))

                    VL[vertex_id] = "L_{}^{}".format(max_var + 1.5, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1


                    VL[vertex_id] = "L_{}^{}".format(max_var + 2, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    VL[vertex_id] = "A_{}^{}".format(max_var + 3, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1


                    edges_to_add.append((LV["A_{}^{}".format(max_var + 1.5, index)], LV["L_{}^{}".format(max_var + 1.5, index)], 0.5, 0))
                    edges_to_add.append((LV["L_{}^{}".format(max_var + 1.5, index)], LV["A_{}^{}".format(max_var + 2, index)], 0.5, 0))
                    edges_to_add.append((LV["A_{}^{}".format(max_var + 2, index)], LV["S_.5^{}".format(index)], 1, 0))
                    edges_to_add.append((LV["A_{}^{}".format(max_var + 2, index)], LV["R_1^{}".format(index)], 1.5, 0))
                    edges_to_add.append((LV["A_{}^{}".format(max_var + 2, index)], LV["L_{}^{}".format(max_var + 2, index)], 0.5, 0))
                    edges_to_add.append((LV["L_{}^{}".format(max_var + 2, index)], LV["A_{}^{}".format(max_var + 3, index)], 0.5, 0))

                else:
                    VL[vertex_id] = "S_{}^{}".format(i, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    VL[vertex_id] = "R_{}^{}".format(i+1, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    VL[vertex_id] = "L_{}^{}".format(max_var + i + 1, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    VL[vertex_id] = "A_{}^{}".format(max_var + i + 2, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    if i == 0:

                        VL[vertex_id] = "A_{}^{}".format(max_var + 1.5, index)
                        LV[VL[vertex_id]] = vertex_id
                        vertex_id += 1
                        edges_to_add.append((LV["R_{}^{}".format(i, index)], LV["S_{}^{}".format(0,index)], 0.5, 0))
                        edges_to_add.append((LV["A_{}^{}".format(max_var+1, index)], LV["R_{}^{}".format(0,index)], 3/2, 0))
                        edges_to_add.append((LV["A_{}^{}".format(max_var+1, index)], LV["L_{}^{}".format(max_var+1,index)], 1/2, 0))
                        edges_to_add.append((LV["L_{}^{}".format(max_var+1, index)], LV["A_{}^{}".format(max_var+1.5,index)], 1/2, 0))
                    else:
                        edges_to_add.append((LV["R_{}^{}".format(i, index)], LV["S_{}^{}".format(i,index)], 2**(i-1), 0))
                        edges_to_add.append((LV["S_{}^{}".format(i, index)], LV["R_{}^{}".format(i+1,index)], 2**i, 0))
                        edges_to_add.append((LV["A_{}^{}".format(max_var +i+1, index)], LV["S_{}^{}".format(i-1,index)], 1, 0))
                        edges_to_add.append((LV["A_{}^{}".format(max_var+i+1, index)], LV["R_{}^{}".format(i,index)], 2**(i-1) + 1, 0))
                        edges_to_add.append((LV["A_{}^{}".format(max_var+i+1, index)], LV["L_{}^{}".format(max_var+i+1,index)], 2**(i-1), 0))
                        edges_to_add.append((LV["L_{}^{}".format(max_var+i+1, index)], LV["A_{}^{}".format(max_var+i+2,index)], 2**(i-1), 0))
                
            
            VL[vertex_id] = "S_{}^{}".format(l, index)
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1
            edges_to_add.append((LV["A_{}^{}".format(max_var+l+1, index)], LV["S_{}^{}".format(l-1,index)], 1, 0))
            edges_to_add.append((LV["A_{}^{}".format(max_var+l+1, index)], LV["R_{}^{}".format(l,index)], 2**(l-1) + 1, 0))
            edges_to_add.append((LV["R_{}^{}".format(l, index)], LV["S_{}^{}".format(l,index)], 2**(l-1), 0))
            edges_to_add.append((LV["S_{}^{}".format(l, index)], LV["T^{}".format(index)], 0, 0))
        elif b[index] > 0 and rel[index] == "==":
            #if the bound is positive and we have an equality constraint we do not add a buffer, we simply add a 0 weight edge from the final partial sum vertex to the target vertex to correctly encode the equality constraint
            edges_to_add.append((LV["P_{}^{}".format(max_var, index)], LV["T^{}".format(index)], 0, 0))
        else:
            #if the bound is negative this will force the error,
            # if the bound is 0 this will correctly force the P_n vertex to be at the P_0 vertex meaning the sum did not extend
            if b[index] == 0:
                edges_to_add.append((LV["P_{}^{}".format(max_var, index)], LV["P_0"], 0, 0))
            else:
                edges_to_add.append((LV["P_{}^{}".format(max_var, index)], LV["T^{}".format(index)], 0, 0)) #the bound is negative so we add a 0 vertex from the final partial sum vertex to the target vertex to force the error
            #finished encoding the buffer

        ########### adding the auxiliary variables #############
        ############ connect the x_1 variable to the partial sum #####################
        if get_literal_coeff(Ax, index, 1) != 0:
            VL[vertex_id] = "B_{}^{}".format(1, index)
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1

            VL[vertex_id] = "C_{}^{}".format(1, index)
            LV[VL[vertex_id]] = vertex_id
            vertex_id += 1
            edges_to_add.append((LV["B_{}^{}".format(1, index)], LV["C_{}^{}".format(1, index)], (get_literal_coeff(Ax, index, 1)/2) - 1, 0))
            edges_to_add.append((LV["C_{}^{}".format(1, index)], LV["P_{}^{}".format(1, index)], (get_literal_coeff(Ax, index, 1)/2) - 1, 0))
            if get_literal_sign(Ax, index, 1):
                #we connect the literal
                edges_to_add.append((LV["X+{}".format(1)], LV["B_{}^{}".format(1, index)], 2, 0))
            else:
                #we connect the negation of the literal
                edges_to_add.append((LV["X-{}".format(1)], LV["B_{}^{}".format(1, index)], 2, 0))

        #connect the rest of the literal
        for j in range(2, max_var+1):
            if get_literal_coeff(Ax, index, j) != 0:
                #only do this for the edges that have a weight larger than 1
                VL[vertex_id] = "B_{}^{}".format(j, index)
                LV[VL[vertex_id]] = vertex_id
                vertex_id += 1

                VL[vertex_id] = "C_{}^{}".format(j, index)
                LV[VL[vertex_id]] = vertex_id
                vertex_id += 1
                
                #the auxiliary literal for x_j in the index's constraint
                VL[vertex_id] = "x_{}^{}".format(j, index)
                LV[VL[vertex_id]] = vertex_id
                vertex_id += 1
                
                ######### mechanism that dictates the extension of the sum from the position of the x_j variable ###################
                edges_to_add.append((LV["x_{}^{}".format(j, index)], LV["A_{}^{}".format(j,index)], 1, 0)) #connect it to the A_j vertex which is behind the P_j-1 vertex
                edges_to_add.append((LV["x_{}^{}".format(j, index)], LV["B_{}^{}".format(j,index)], 2, 0))
                edges_to_add.append((LV["B_{}^{}".format(j, index)], LV["C_{}^{}".format(j,index)], (get_literal_coeff(Ax, index, j)/2) - 1, 0))
                edges_to_add.append((LV["C_{}^{}".format(j, index)], LV["P_{}^{}".format(j,index)], (get_literal_coeff(Ax, index, j)/2) - 1, 0))


            
            ########### ENCODING THE LITERAL EXTENSION MECHANISM ##################################
                VL[vertex_id] = "N_({}. {})^{}".format(j, 1, index)
                LV[VL[vertex_id]] = vertex_id
                vertex_id += 1

                if get_literal_sign(Ax, index, j):
                    #connect it to the the literal
                    edges_to_add.append((LV["X+{}".format(j)], LV["N_({}. {})^{}".format(j, 1,index)], get_literal_coeff(Ax, index, 1)/2, 0))
                else:
                    #connect it to the negated literal, 
                    edges_to_add.append((LV["X-{}".format(j)], LV["N_({}. {})^{}".format(j, 1,index)], get_literal_coeff(Ax, index, 1)/2, 0))

                for k in range(1, j-1):
                    VL[vertex_id] = "Q_({}. {})^{}".format(j, k, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    VL[vertex_id] = "N_({}. {})^{}".format(j, k+1, index)
                    LV[VL[vertex_id]] = vertex_id
                    vertex_id += 1

                    edges_to_add.append((LV["N_({}. {})^{}".format(j, k, index)], LV["Q_({}. {})^{}".format(j, k,index)], get_literal_coeff(Ax, index, k)/2, 0))
                    edges_to_add.append((LV["Q_({}. {})^{}".format(j, k, index)], LV["N_({}. {})^{}".format(j, k+1,index)], get_literal_coeff(Ax, index, k+1)/2, 0))
                
                edges_to_add.append((LV["N_({}. {})^{}".format(j, j-1, index)], LV["x_{}^{}".format(j,index)], get_literal_coeff(Ax, index, j-1)/2, 0))

        E.update(edges_to_add)
        c2e[index] = edges_to_add
    return (E, VL, LV, c2e, max_var)

def contract_zero_edges(E, VL, LV):

    # --- DSU structure ---
    parent = {v: v for v in VL}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # --- merge vertices connected by zero edges ---
    for (u, v, c, I) in E:
        if abs(c) < 1e-12:
            union(u, v)

    # --- build representative mapping ---
    rep = {v: find(v) for v in VL}

    # --- rebuild VL and LV ---
    new_VL = {}
    new_LV = {}
    rep_to_new = {}

    new_id = 1
    for v in sorted(set(rep.values())):
        rep_to_new[v] = new_id
        name = VL[v]
        new_VL[new_id] = name
        new_LV[name] = new_id
        new_id += 1

    # --- rebuild edges ---
    new_E = set()

    for (u, v, c, I) in E:
        if abs(c) < 1e-12:
            continue

        ru = rep_to_new[rep[u]]
        rv = rep_to_new[rep[v]]

        if ru != rv:
            if ru > rv:
                ru, rv = rv, ru
            new_E.add((ru, rv, c, I))

    return new_E, new_VL, new_LV

#################### Main ###########################
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python blp2dgp.py <inputfile.opb>")
        sys.exit(1)
    opbf = sys.argv[1] #the first argument given to the program
    opbbn = os.path.basename(opbf)  # basename
    dgpf = ''.join(opbbn.split('.')[:-1]) + "-dgp.dat"  # output DGP to AMPL .dat
    blp2dgpf = ''.join(opbbn.split('.')[:-1]) + "-blp2dgp.pkl"  # output DGP to AMPL .dat

    blp_instance = readOpb(opbf)

    (E, VL, LV, c2e, max_var) = reduce_blp_2_dgp(blp_instance)

    E, VL, LV = contract_zero_edges(E, VL, LV)



    ## make a graph G=(V,E)
    E = sorted(list(E))
    V = [vid for vid in VL]
    V.sort()
    G = (V,E,VL,max_var)

    ## write DGP1 instance
    writeDat(G, dgpf, opbf)

    ## write c2e dictionary
    with open(blp2dgpf, 'wb') as f:
        pickle.dump(c2e, f)

