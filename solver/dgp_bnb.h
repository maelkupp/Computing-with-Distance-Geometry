//The header file for all the functions used in my MinErr BnB solver
#include "gurobi_c++.h"
#include <tuple>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <string>
#include <iostream>
#include <set>
#include <memory>

// 0 ---------------------- Structs used ------------------------------------------------------
constexpr double PI = 3.14159265358979323846;


class Point{
    public:
        double x;
        double y;
        //defualt constructor and constructor
        Point(): x(0), y(0) {}
        Point(double x, double y): x(x), y(y) {}
        std::string display() const;

};


struct Edge{
    int u; //id of 1st vertex so if there is a directed edge this is always the source node
    int v; //if of 2nd vertex
    double weight; //the weight of the edge
    bool directed; //whether it is a directed edge


    Edge() = default;
    //simple constructor for now
    Edge(int u, int v, double weight, bool directed): u(u), v(v), weight(weight), directed(directed){}

    void display() const {
        std::string directed_string = "";
        if(directed){
            directed_string = "Directed";
        }
        std::cerr << directed_string << "Edge between vertices u: " << u << " and v: " << v << " of weight " << weight << "\n";
    }

};


enum class WeightMode {
    INTEGER,
    REAL_SCALED
};

struct SolverConfig {
    WeightMode mode;
    int scaling_power;   // used only if REAL_SCALED
};

struct IndexedEdge{
    int id;
    int u;
    int v;
    double weight;      // original
    int weight_int;     // scaled version
    int directed;
};

struct CycleID{
    std::vector<int> edge_ids;
    std::vector<int> coeff;   // +1 or -1
};


struct Adjacency {
    int neighbourId;
    double dist;
    bool directed;   // is it a direction constraint? if directed if false we can ignore outgoing and do not initialize it for efficiency
    bool outgoing;   // is the edge outgoing (true) or incoming (false) with respect to the neighbour of this adjacent node

    //if there are three parameters to the contstructor we know it is directed
    Adjacency(int neighbourId, double dist, bool outgoing): neighbourId(neighbourId), dist(dist), directed(true), outgoing(outgoing){} 

    //if two parameters we know it is undirected edge and so do not initialise the outgoing attribute
    Adjacency(int neighbourId, double dist):  neighbourId(neighbourId), dist(dist), directed(false) {}

    void display(){
        std::string directed_string = "";
        if(directed){
            directed_string += "Directed";
        }
        std::cerr << directed_string << " Neighbour with id " << neighbourId << ", " <<  dist << " away \n";
    }

};

// struct that allows me to hash a pair of integers
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

//struct that allows us to easily pair cycle with their minimum errors (weight) used in the greedy_packing_cycle function to obtain as tight a lower bound as possible
struct WeightedCycle{
    std::vector<Edge> edges;
    double error;
};

struct WeightedCycleID{
    int cycle_idx;
    double error;
    std::vector<int> edge_ids;
};

// 1 --------------- Graph + Preprocessing -----------------

//reads a dat file and return a tuple  (vector of edges, set of vertex ids, map from vertex ids -> vertex names)
std::tuple<std::vector<Edge>, std::set<int>, std::unordered_map<int, std::string>> parse_dgp_instance_dat_file(const std::string& file_path);

//creates an adjacency list from a vector of edges of a graph
std::vector<std::vector<Adjacency>> create_adjacency_list_from_edges(
    const std::vector<Edge>& edges,
    const int num_vertices
);

std::vector<IndexedEdge> build_indexed_edges(
    const std::vector<Edge>& edges,
    int scale

);

//creates a spanning forest of a graph from its adjacency list
std::vector<std::vector<Edge>> get_spanning_forest(
    std::vector<std::vector<Adjacency>>& adj_list
);

//creates a cycle basis for each connected component of the graph from a vector of spanning forests (each spanning forest is a spanning forest of a C.C) and adjacency list
std::vector<std::vector<std::vector<Edge>>> get_cycle_basis(
    std::vector<std::vector<Edge>>& spanning_forest_edges,
    std::vector<std::vector<Adjacency>>& adj_list
);

//converts and flattens a cycle basis of Edges into a vector of CycleIDs,
std::vector<CycleID> convert_cycles_to_edge_ids(
    const std::vector<std::vector<std::vector<Edge>>>& cycle_basis,
    const std::vector<IndexedEdge>& indexed_edges
);






// 2 ---------------- Relaxation + Lower Bound system ---------------------

//obtains the minimum error of a cycle using DP
// should implement the version for real edge weights shortly, that is also super important so my solver can do both real & integer edge weights
double DP_cycle_error(const std::vector<double>& cycle);

//performs DP programming on a cycle to obtain the minimum error but some edge orientations are fixed, minimizes only over free edge signs
double DP_cycle_error_with_fixed_signs(
    const CycleID& cycle,
    const std::vector<int>& fixed_signs,
    const std::vector<IndexedEdge>& edges
);

// performs greedy packing on edge disjoint cycles based on their error on a cycle basis
double greedy_packing_cycle_err(
    const std::vector<std::vector<std::vector<Edge>>>& cycle_basis,
    const std::vector<std::vector<double>>& cycle_errors);

//function to compute the LB at each node of the BnB
double compute_cycle_packing_LB(
    const std::vector<CycleID>& cycles,
    const std::vector<double>& cycle_errors,
    const std::vector<IndexedEdge>& edges,
    int scale
);




// 3 ------------------ Upper bound system -----------------------------

//performs coefficient descent on the R line (need an exact explanation of what this function does, leo told me there was an actual name for it)
double coefficient_descent_on_line(
    const std::vector<Edge>& edges,
    const std::vector<std::vector<Adjacency>>& adj_list,
    std::vector<double>& t);

//heuristic MinErrDGP solver, provides a tight upper bound to the error quickly, call the coefficient gradient descent function
std::pair<double, std::vector<double>> optimized_projection_minErrDGP1_UB(
    const std::vector<Edge>& edges,
    const std::set<int>& vertex_ids,
    std::vector<std::vector<Adjacency>>& adj_list

);

//a wrapper for our heuristic defined just above
double compute_initial_UB(
    const std::vector<Edge>& edges,
    const std::set<int>& vertex_ids
);




// 4 ------------------ Exact embedding solver -------------
//this is the part at the leaves of the tree, where we compute min || Bx - d \odot s ||

// ExactEmbeddingSolver.h

class ExactEmbeddingSolver {
private:
    GRBEnv env;
    std::unique_ptr<GRBModel> model;   // Must be a pointer!

    std::vector<GRBVar> x;
    std::vector<GRBVar> t;
    std::vector<GRBConstr> c1;
    std::vector<GRBConstr> c2;

    const std::vector<IndexedEdge>& edges;

public:
    ExactEmbeddingSolver(const std::vector<IndexedEdge>& edges_, int n_vertices);
    std::pair<double, std::vector<double>> solve(const std::vector<int>& fixed_signs);
};

//wrapper that does all the steps in computing the exact embedding that minimizes the error for a fixed sign, is what guarantees the correctness of the BnB
double solve_exact_embedding(
    const std::vector<int>& fixed_signs
);

// 5 ----------------------- Branch and Bound Engine -----------------------

//selects the edge we will branch on, for now keep it simple, will improve it later on (edges in most cycles, edge in largest error cycle)

int select_branching_edge(
    const std::vector<int>& fixed_signs
);

//the recursive function for our branch and bound
void branch_and_bound(
    ExactEmbeddingSolver& exact_solver,
    const std::vector<CycleID>& i_cycles,
    const std::vector<IndexedEdge>& i_edges,
    const std::vector<std::vector<int>>& edge_to_cycles,
    std::vector<int>& fixed_signs,
    double& bestUB,
    std::vector<double>& bestEmbedding,
    int scale
);

//solver wrapper, will see what the exact signature for this function is in bit
std::pair<double, std::vector<double>> solve_minerr_dgp1(const std::string& filename, int weight_flag);