#include "dgp_bnb.h"
#include "random.h"
#include "gurobi_c++.h"
#include <tuple>
#include <vector>
#include <climits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stack>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <numeric>


// 1 ---------------------- Graph + Preprocessing
std::tuple<
    std::vector<Edge>,
    std::set<int>,
    std::unordered_map<int,std::string>
>
parse_dgp_instance_dat_file(const std::string& filepath)
{
    std::ifstream infile(filepath);
    if (!infile)
        throw std::runtime_error("Cannot open file: " + filepath);

    std::string line;
    bool in_edge_block = false;

    std::vector<std::tuple<int,int,double,int>> raw_edges;
    std::set<int> raw_vertex_ids;
    std::unordered_map<int,std::string> raw_vertex_names;

    while (std::getline(infile, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        // trim
        auto l = line.find_first_not_of(" \t");
        if (l == std::string::npos) continue;
        auto r = line.find_last_not_of(" \t");
        line = line.substr(l, r - l + 1);

        size_t hash_pos = line.find('#');

        std::string comment_part;
        if (hash_pos != std::string::npos)
        {
            comment_part = line.substr(hash_pos + 1);
            line = line.substr(0, hash_pos);
        }

        if (line.empty()) continue;

        if (!in_edge_block)
        {
            std::string lower = line;
            std::transform(lower.begin(), lower.end(),
                           lower.begin(), ::tolower);

            if (lower.find("param") != std::string::npos &&
                lower.find("e") != std::string::npos &&
                lower.find(":=") != std::string::npos)
            {
                in_edge_block = true;
            }
            continue;
        }

        if (line.find(';') != std::string::npos)
        {
            in_edge_block = false;
            continue;
        }

        std::stringstream ss(line);

        int u, v;
        double dist;
        int flag = 0;

        if (!(ss >> u >> v >> dist))
            continue;

        ss >> flag;

        raw_edges.emplace_back(u, v, dist, flag);
        raw_vertex_ids.insert(u);
        raw_vertex_ids.insert(v);

        // -------- Parse vertex names from comment --------
        if (!comment_part.empty())
        {
            auto lb = comment_part.find('[');
            auto rb = comment_part.find(']');

            if (lb != std::string::npos && rb != std::string::npos && rb > lb)
            {
                std::string inside = comment_part.substr(lb + 1, rb - lb - 1);

                auto comma = inside.find(',');

                if (comma != std::string::npos)
                {
                    std::string name_u = inside.substr(0, comma);
                    std::string name_v = inside.substr(comma + 1);

                    // trim spaces
                    auto trim = [](std::string& s)
                    {
                        auto l = s.find_first_not_of(" \t");
                        auto r = s.find_last_not_of(" \t");
                        if (l == std::string::npos) { s=""; return; }
                        s = s.substr(l, r-l+1);
                    };

                    trim(name_u);
                    trim(name_v);

                    raw_vertex_names[u] = name_u;
                    raw_vertex_names[v] = name_v;
                }
            }
        }
    }

    if (raw_edges.empty())
        throw std::runtime_error("No edges found in file.");

    // ---- Relabel vertices ----
    std::unordered_map<int,int> relabel;
    int new_id = 1;

    for (int vid : raw_vertex_ids)
        relabel[vid] = new_id++;

    std::vector<Edge> edges;
    std::set<int> vertex_ids;

    for (const auto& [u,v,dist,flag] : raw_edges)
    {
        int nu = relabel[u];
        int nv = relabel[v];

        if(flag == 1)
            edges.emplace_back(nu, nv, dist, flag);
        else
            edges.emplace_back(std::min(nu,nv), std::max(nu,nv), dist, flag);

        vertex_ids.insert(nu);
        vertex_ids.insert(nv);
    }

    // ---- Remap vertex names ----
    std::unordered_map<int,std::string> vertex_id_to_name;

    for (const auto& [old_id,name] : raw_vertex_names)
        vertex_id_to_name[relabel[old_id]] = name;

    return {edges, vertex_ids, vertex_id_to_name};
}

std::vector<std::vector<Adjacency>> create_adjacency_list_from_edges(const std::vector<Edge>& edges, int num_vertices){
    //function that creates an adjacency list of a DGP instance from a vector of Edges
    std::vector<std::vector<Adjacency>> adj_list(num_vertices);

    for(const Edge& edge: edges){
        if(edge.directed){
            adj_list[edge.u-1].push_back(Adjacency(edge.v, edge.weight, true));
            adj_list[edge.v-1].push_back(Adjacency(edge.u, edge.weight, false));
        }else{
            adj_list[edge.u-1].push_back(Adjacency(edge.v, edge.weight));
            adj_list[edge.v-1].push_back(Adjacency(edge.u, edge.weight));    
        }
    }

    return adj_list;
}


std::vector<IndexedEdge> build_indexed_edges(
    const std::vector<Edge>& edges,
    int scale
){
    std::vector<IndexedEdge> indexed_edges;
    indexed_edges.reserve(edges.size());

    for(int i=0; i< static_cast<int>(edges.size()); ++i)
    {

        int w_int = static_cast<int>(std::round(edges[i].weight * scale));

        indexed_edges.push_back({
            i,
            edges[i].u,
            edges[i].v,
            edges[i].weight,
            w_int,
            edges[i].directed
        });
    }

    return indexed_edges;
}


std::pair<std::vector<Edge>, std::unordered_set<int>> dfs_for_spanning_tree(
    std::vector<std::vector<Adjacency>>& adj_list, 
    int start_vertex
){
    //returns a pair, a vector of Edges in the spanning tree of the connected component and a set of all the vertices it visitied during the DFS
    std::unordered_set<int> visited;
    std::vector<Edge> tree_edges;

    std::stack<std::tuple<int, int, double>> stk; // (parent, current)
    stk.push({-1, start_vertex, 0.0});
    visited.insert(start_vertex);

    while (!stk.empty()) {
        auto [parent, u, dist] = stk.top();
        stk.pop();

        if (parent != -1) {
            // find distance from parent -> u, add edge to the vector of edges of the spanning tree
            tree_edges.push_back({parent, u, dist, 0});

        }
        
        //look through each neighbour in the adjacency of u, if we have not yet visited the node, we mark it as visited and and the edge to the stack
        for (const Adjacency& adj : adj_list[u-1]) {
            int v = adj.neighbourId;
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                stk.push({u, v, adj.dist});
            }
        }
    }

    return {tree_edges, visited};
}

std::vector<std::vector<Edge>> get_spanning_forest(std::vector<std::vector<Adjacency>>& adj_list){
    //this function returns a vector of edges (all assumed to be undirected) corresponding to a spanning tree of a graph represented with its
    //adjacency list, this is done by performing a DFS through the tree
    //if there are multiple connected components in our graph this will return a spanning forest
    
    std::vector<std::vector<Edge>> spanning_forest {};
    std::unordered_set<int> visited {};

    for(int i=0; i<static_cast<int>(adj_list.size()); ++i){
        int v_id = i+1;
        //looks if we have visited this vertex, if we have not, we are looking at a new connected component of the graph
        if(visited.find(v_id) == visited.end()){
            //we have not visited the ID yet
            auto tree_pair = dfs_for_spanning_tree(adj_list, v_id);
            std::vector<Edge> spanning_tree = tree_pair.first;
            std::unordered_set<int> visited_tree = tree_pair.second;


            //update the visited set, sets all the vertices in the connected component as visited
            spanning_forest.push_back(spanning_tree);
            for(const auto& visited_id: visited_tree){
                visited.insert(visited_id);
            }
        }
    }
    return spanning_forest;
}

//function that creates a standardised edge set for an edge (u,v) -> (min(u,v), max(u,v))
std::unordered_set<std::pair<int, int>, PairHash> create_tree_edge_set(const std::vector<Edge>& spanning_tree_edges){
    std::unordered_set<std::pair<int, int>, PairHash> tree_edges_set{};
    for(const auto& edge: spanning_tree_edges){
        tree_edges_set.insert({std::min(edge.u, edge.v), std::max(edge.u, edge.v)});
    }
    return tree_edges_set;
}

std::pair<std::unordered_map<int, Edge>,std::unordered_map<int, int>>get_parent_and_depth_maps_DFS(
    std::vector<std::vector<Adjacency>>& adj_list,
    std::unordered_set<std::pair<int, int>, PairHash>& tree_edge_set,
    int root_id
){
    // parent_edge[v] = edge connecting v to its parent in the DFS tree
    std::unordered_map<int, Edge> parent_edge;
    //depth[v] = depth of the vertex v in the DFS tree
    std::unordered_map<int, int> depth;

    depth[root_id] = 0;
    // root has no parent edge

    std::stack<int> stk;
    std::unordered_set<int> visited;

    visited.insert(root_id);
    stk.push(root_id);

    while (!stk.empty()) {
        int u = stk.top();
        stk.pop();

        for (const auto& adj : adj_list[u-1]) {
            int v = adj.neighbourId;

            // Only follow tree edges
            auto key = std::make_pair(std::min(u,v), std::max(u,v));
            if(!tree_edge_set.count(key))
                continue;

            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                stk.push(v);

                // store the actual parent edge
                parent_edge[v] = Edge(u, v, adj.dist, 0);
                depth[v] = depth[u] + 1;
            }
        }
    }

    return {parent_edge, depth};
}



std::vector<std::vector<std::vector<Edge>>>
get_cycle_basis(
    std::vector<std::vector<Edge>>& spanning_forest_edges,
    std::vector<std::vector<Adjacency>>& adj_list
){
    std::vector<std::vector<std::vector<Edge>>> cycle_basis;

    auto find_weight_in_adj = [&](int a, int b) -> double {
        // assumes adjacency list stores both directions
        for(const auto& adj : adj_list[a-1]){
            if(adj.neighbourId == b) return adj.dist;
        }
        return 0.0;
    };

    for(const auto& spanning_tree_edges : spanning_forest_edges)
    {
        if(spanning_tree_edges.empty()) continue;

        std::unordered_set<std::pair<int,int>, PairHash> tree_edge_set =
            create_tree_edge_set(spanning_tree_edges);

        auto [parent_edge_tree, depth_tree] =
            get_parent_and_depth_maps_DFS(
                adj_list,
                tree_edge_set,
                spanning_tree_edges[0].u
            );

        std::vector<std::vector<Edge>> component_cycle_basis;

        for(int i=0; i<static_cast<int>(adj_list.size()); ++i)
        {
            for(const auto& adj : adj_list[i])
            {
                int v = adj.neighbourId;
                int u = i+1;


                // avoid duplicates (undirected edge handled once)
                if(v >= u) continue;

                // skip if not in this component
                if(parent_edge_tree.find(u) == parent_edge_tree.end() ||
                   parent_edge_tree.find(v) == parent_edge_tree.end())
                    continue;

                // skip tree edges
                if(tree_edge_set.count({std::min(u,v), std::max(u,v)}))
                    continue;

                // ---------- build fundamental cycle as vertex path ----------

                int x = u;
                int y = v;

                // vertex paths from u->... and v->...
                std::vector<int> path_u; // starts at u, goes up
                std::vector<int> path_v; // starts at v, goes up

                path_u.push_back(x);
                path_v.push_back(y);

                // lift to same depth
                while(depth_tree[x] > depth_tree[y]) {
                    const Edge& pe = parent_edge_tree.at(x);
                    int p = (pe.u == x ? pe.v : pe.u);
                    x = p;
                    path_u.push_back(x);
                }

                while(depth_tree[y] > depth_tree[x]) {
                    const Edge& pe = parent_edge_tree.at(y);
                    int p = (pe.u == y ? pe.v : pe.u);
                    y = p;
                    path_v.push_back(y);
                }

                // lift both to LCA
                while(x != y) {
                    {
                        const Edge& pe = parent_edge_tree.at(x);
                        int p = (pe.u == x ? pe.v : pe.u);
                        x = p;
                        path_u.push_back(x);
                    }
                    {
                        const Edge& pe = parent_edge_tree.at(y);
                        int p = (pe.u == y ? pe.v : pe.u);
                        y = p;
                        path_v.push_back(y);
                    }
                }

                // Now x==y is LCA.
                // path_u: u -> ... -> lca
                // path_v: v -> ... -> lca

                // Build ordered vertex cycle:
                // u -> ... -> lca -> ... -> v -> u
                std::vector<int> cycle_vertices = path_u;

                // append reverse(path_v) but skip the lca duplicate
                for(int i = (int)path_v.size() - 2; i >= 0; --i) {
                    cycle_vertices.push_back(path_v[i]);
                }

                // close cycle by returning to start u
                cycle_vertices.push_back(u);

                // Convert to directed edges in that order
                std::vector<Edge> cycle_edges;
                cycle_edges.reserve(cycle_vertices.size() - 1);

                for(size_t i = 0; i + 1 < cycle_vertices.size(); ++i){
                    int a = cycle_vertices[i];
                    int b = cycle_vertices[i+1];

                    double w = find_weight_in_adj(a, b); // works for tree edges and the non-tree closure edge
                    cycle_edges.push_back(Edge(a, b, w, 0));
                }

                component_cycle_basis.push_back(std::move(cycle_edges));
            }
        }

        if(!component_cycle_basis.empty())
            cycle_basis.push_back(std::move(component_cycle_basis));
    }

    return cycle_basis;
}

std::vector<CycleID> convert_cycles_to_edge_ids(
    const std::vector<std::vector<std::vector<Edge>>>& cycle_basis,
    const std::vector<IndexedEdge>& indexed_edges
){
    std::vector<CycleID> indexed_cycles;

    std::unordered_map<std::pair<int,int>, int, PairHash> edge_map;

    // map canonical (min,max) -> edge id
    for(const auto& ie : indexed_edges){
        std::pair<int,int> key = {
            std::min(ie.u, ie.v),
            std::max(ie.u, ie.v)
        };
        edge_map[key] = ie.id;
    }

    for(const auto& cc_basis : cycle_basis){
        for(const auto& cycle : cc_basis){

            CycleID cid;

            for(const auto& e : cycle){

                std::pair<int,int> key = {
                    std::min(e.u, e.v),
                    std::max(e.u, e.v)
                };

                int e_id = edge_map[key];

                const auto& stored = indexed_edges[e_id];

                // determine orientation
                int coeff = 0;

                if(stored.u == e.u && stored.v == e.v){
                    coeff = +1;
                }
                else{
                    coeff = -1;
                }

                cid.edge_ids.push_back(e_id);
                cid.coeff.push_back(coeff);
            }

            indexed_cycles.push_back(cid);
        }
    }

    return indexed_cycles;
}



// 2 ---------------- Relaxation + Lower Bound system ---------------------


double DP_cycle_error_with_fixed_signs(
    const CycleID& cycle,
    const std::vector<int>& fixed_signs,
    const std::vector<IndexedEdge>& edges
){
    long long fixed_sum = 0;
    std::vector<int> free_weights;

    for(size_t k = 0; k < cycle.edge_ids.size(); ++k){
        int e_id = cycle.edge_ids[k];
        int sigma = cycle.coeff[k];
        const auto& e = edges[e_id];

        if(fixed_signs[e_id] != 0){
            fixed_sum += sigma * fixed_signs[e_id] * e.weight_int;
        } else {
            free_weights.push_back((int)std::abs(e.weight_int));
        }
    }

    int S = std::accumulate(free_weights.begin(),
                            free_weights.end(), 0);

    if (S == 0)
        return std::abs(fixed_sum);
    if (S > 5'000'000){
        throw std::runtime_error("Cycle sum too large for DP");
    }

    const int WORD = 64;
    int num_words = (S + WORD) / WORD;

    std::vector<uint64_t> dp(num_words, 0);
    dp[0] = 1ULL;   // sum 0 reachable

    for(int w : free_weights)
    {
        int word_shift = w / WORD;
        int bit_shift  = w % WORD;

        std::vector<uint64_t> shifted(num_words, 0);

        for(int i = num_words - 1; i >= 0; --i)
        {
            uint64_t val = dp[i];
            if(!val) continue;

            int target = i + word_shift;
            if(target >= num_words) continue;

            shifted[target] |= (val << bit_shift);

            if(bit_shift && target + 1 < num_words)
                shifted[target + 1] |= (val >> (WORD - bit_shift));
        }

        for(int i = 0; i < num_words; ++i)
            dp[i] |= shifted[i];
    }

    double best_val = std::numeric_limits<double>::infinity();

    for(int p = 0; p <= S; ++p)
    {
        if(dp[p / WORD] & (1ULL << (p % WORD)))
        {
            double val = std::abs(fixed_sum + 2*p - S);
            if(val < best_val){
                best_val = val;
            }
        }
    }

    return best_val;
}


// double DP_cycle_error_with_fixed_signs(
//     const CycleID& cycle,
//     const std::vector<int>& fixed_signs,
//     const std::vector<IndexedEdge>& edges
// ){

//     double fixed_sum {0.0};
//     std::vector<double> free_weights {};

//     for(size_t k = 0; k < cycle.edge_ids.size(); ++k){

//         int e_id = cycle.edge_ids[k];
//         int sigma = cycle.coeff[k];     // ← ORIENTATION

//         const auto& e = edges[e_id];


//         if(fixed_signs[e_id] != 0){
//             fixed_sum += sigma * fixed_signs[e_id] * e.weight;
//         }else{
//             // free variable contributes ± sigma * weight
//             free_weights.push_back(std::abs(e.weight));
//         }
//     }


//     int S = std::accumulate(free_weights.begin(), free_weights.end(), 0);

//     std::vector<char> dp(S + 1, 0);
//     dp[0] = 1;

//     for(int w : free_weights){
//         for(int j = S - w; j >= 0; --j){
//             if(dp[j]) dp[j + w] = 1;
//         }
//     }


//     int best = 0;

//     for(int p = 0; p <= S; ++p){
//         if(!dp[p]) continue;

//         double val = std::abs(fixed_sum + 2*p - S);
//         double best_val = std::abs(fixed_sum + 2*best - S);


//         if(val < best_val)
//             best = p;
//     }

//     double cycle_error = std::fabs(fixed_sum + 2*best - S);


//     return cycle_error;
// }



std::vector<double> get_cycle_basis_error(
    const std::vector<CycleID>& indexed_cycles,
    const std::vector<IndexedEdge>& i_edges
){
    std::vector<double> cycle_basis_error(static_cast<int>(indexed_cycles.size()), 0); //we already know the size
    std::vector<int> fixed_signs = std::vector<int>(i_edges.size(), 0);
    for(int i=0; i<static_cast<int>(indexed_cycles.size()); ++i){
        cycle_basis_error[i] = DP_cycle_error_with_fixed_signs(indexed_cycles[i], fixed_signs, i_edges);
    }

    return cycle_basis_error;
}

//this function takes a cycle basis and the associated error (weight) of that cycle, the point is to obtain the best possible error
//by greedily selecting the cycle that are edge disjoint and have the greatest error

double compute_cycle_packing_LB(
    const std::vector<CycleID>& cycles,
    const std::vector<int>& fixed_signs,
    const std::vector<IndexedEdge>& edges,
    int scale
){

    int n = static_cast<int>(cycles.size());
    std::vector<WeightedCycleID> vec_weighted_cycle;
    vec_weighted_cycle.resize(n);
    for(int i=0; i<n; ++i){
        vec_weighted_cycle[i] = {i, DP_cycle_error_with_fixed_signs(cycles[i], fixed_signs, edges)/scale, cycles[i].edge_ids};
    }

    //sort in decreasing order by weight (to have the cycles contributing to the error the most at the beginning)
    std::sort(vec_weighted_cycle.begin(), vec_weighted_cycle.end(),
            [](const WeightedCycleID& a, const WeightedCycleID& b) {
                return a.error > b.error;
            });

    std::vector<bool> used_edges(edges.size(), false);
    double total_lb = 0.0;

    for (const auto& cycle : vec_weighted_cycle) {

        bool disjoint = true;

        for (int e_id : cycle.edge_ids) {
            if (used_edges[e_id]) {
                disjoint = false;
                break;
            }
        }

        if (disjoint) {
            total_lb += cycle.error;

            for (int e_id : cycle.edge_ids) {
                used_edges[e_id] = true;
            }
        }
    }

    return total_lb;
}


// 3 ------------------ Upper bound system -----------------------------

double local_obj_abs(
    double x,
    int i_v,
    const std::vector<std::vector<Adjacency>>& adj_list,
    const std::vector<double>& t
){
    double sum = 0.0;
    const auto& nbrs = adj_list[i_v];
    for (const auto& nbr : nbrs) {
        int j = nbr.neighbourId - 1;   // 0-based index into t
        sum += std::abs(std::abs(x - t[j]) - nbr.dist);
    }
    return sum;
}


double coefficient_descent_on_line(const std::vector<Edge>& edges,
    const std::vector<std::vector<Adjacency>>& adj_list,
    std::vector<double>& t
){  

    //callable lambda to obtain the error of an embedding
    auto global_error = [&](const std::vector<double>& tt){
        double E = 0.0;
        for (const auto& e : edges) {
            E += std::abs(std::abs(tt[e.u - 1] - tt[e.v - 1]) - e.weight);
        }
        return E;
    };

    double prev_E = global_error(t);

    //best constants I have found with some testing, the testing what not very rigorous, can be improved upon
    const double lambda = 0.5;
    const double rel_tol = 5e-5;
    const int max_sweeps = 100; //still have a max sweep to prevent non - convergence


    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        // Jacobi-style buffer for stability
        std::vector<double> t_new = t;
        //have 1 -indexed for the vertex ids
        for (int i = 0; i < static_cast<int>(t.size()); ++i) {
            //int vid = i-1;
            const std::vector<Adjacency>& ngbrs = adj_list[i];
            if(ngbrs.size() == 0){
                //if for some reason the vertex is not in the adjacency list keys or it does not have any neighbours continue to the next vertex
                continue;
            }

            // Build candidate set: {t_j ± d_ij} plus current t_i
            std::vector<double> cand;
            cand.reserve(2 * ngbrs.size() + 1);
            cand.push_back(t[i]);

            for (const auto& nbr : ngbrs){
                int j = nbr.neighbourId - 1;
                double dij = nbr.dist;

                cand.push_back(t[j] + dij);

                if(!nbr.directed)
                    cand.push_back(t[j] - dij);
                else
                    cand.back() = nbr.outgoing ? t[j] + dij : t[j] - dij;
            }

            // Pick best candidate for the TRUE local objective
            double best_x = cand[0];
            double best_val = local_obj_abs(best_x, i, adj_list, t);
            for (int k = 1; k < (int)cand.size(); ++k) {
                double val = local_obj_abs(cand[k], i, adj_list, t);
                if (val < best_val) {
                    best_val = val;
                    best_x = cand[k];
                }
            }

            // Damped update
            t_new[i] = (1.0 - lambda) * t[i] + lambda * best_x; //move part of the way towards the locally optimal position
            //if we jump all the way we might have strong reactions to conflicting constraints etc, often ping - pong back and forth
        }

        t.swap(t_new); // same as t = t_new but faster

        double curr_E = global_error(t);
        double rel_impr = std::abs(prev_E - curr_E) / std::max(1.0, prev_E);

        if (rel_impr < rel_tol) {
            break;
        }

        prev_E = curr_E;


    }

    double Efinal = global_error(t);
    return Efinal;
}


std::pair<double, std::vector<double>> optimized_projection_minErrDGP1_UB(
    const std::vector<Edge>& edges,
    const std::set<int>& vertex_ids,
    std::vector<std::vector<Adjacency>>& adj_list
){
    //this function creates a UB for the MinerrDGP1 using random scatterings of the points onto the line and then performing the coefficient descent algorithm
    // Build adjacency list once

    const int n = static_cast<int>(vertex_ids.size());
    if (n == 0) return {0.0, {}};

    // ---- estimate a reasonable scale from the edges ----
    double avg_d = 0.0;
    for (const auto& e : edges) avg_d += e.weight;
    avg_d /= std::max(1, (int)edges.size());

    const int num_trials = 10;          // number of random restarts
    const double spread = 2.0 * avg_d;  // scale of random scattering
    const double noise = 0.1 * avg_d;   // small perturbation

    double best_error = std::numeric_limits<double>::infinity();
    std::vector<double> t_best(n, 0);

    // Convert vertex_ids set to vector for indexing
    std::vector<int> vids(vertex_ids.begin(), vertex_ids.end());

    for (int trial = 0; trial < num_trials; ++trial) {

        // ---- initialize t ----
        std::vector<double> t(n, 0.0);

        // Pick a random root
        int root_idx = Random::get(0, n - 1);
        int root_vid = vids[root_idx];
        t[root_idx] = 0.0;

        // Simple BFS-style initialization
        std::unordered_map<int, bool> visited;
        visited[root_vid] = true;

        std::vector<int> stack;
        stack.push_back(root_vid);

        while (!stack.empty()) {
            int u = stack.back();
            stack.pop_back();

            int u_idx = u-1;

            for (const auto& nbr : adj_list[u-1]) {
                int v = nbr.neighbourId;
                if (visited[v]) continue;

                int v_idx = v-1;

                // random sign
                double sign = (Random::get(0, 1) == 0) ? -1.0 : 1.0;
                t[v_idx] = t[u_idx] + sign * nbr.dist;

                // small noise to avoid symmetry lock-in
                t[v_idx] += Random::get<double>(-noise, noise);

                visited[v] = true;
                stack.push_back(v);
            }
        }

        // For disconnected vertices, scatter randomly, should change this to scatter points of each connected component, here we are only doing it for one connected component
        for (int i = 0; i < n; ++i) {
            if (!visited[vids[i]]) {
                t[i] = Random::get<double>(-spread, spread);
            }
        }

        // ---- run the 1D solver ----
        double err = coefficient_descent_on_line(edges, adj_list, t);

        if(err < best_error){
            best_error = err;
            t_best = t;
        }
    }

    return {best_error, t_best};
}



// 4 ------------------ Exact embedding solver -------------

ExactEmbeddingSolver::ExactEmbeddingSolver(
    const std::vector<IndexedEdge>& edges_,
    int n_vertices
)
    : env(true),
      edges(edges_)
{
    try
    {
        // ==== START the environment BEFORE creating the model ====
        env.set(GRB_IntParam_OutputFlag, 0);
        env.start();

        // Now it is safe to create the model
        model = std::make_unique<GRBModel>(env);

        int m = edges.size();

        // --- Create x variables ---
        x.resize(n_vertices);
        for (int i = 0; i < n_vertices; ++i)
        {
            x[i] = model->addVar(-GRB_INFINITY, GRB_INFINITY,
                                 0.0, GRB_CONTINUOUS);
        }

        // Fix translation
        model->update();
        model->addConstr(x[0] == 0.0);

        // --- Create t variables ---
        t.resize(m);
        c1.resize(m);
        c2.resize(m);

        for (int e = 0; e < m; ++e)
        {
            int u = edges[e].u - 1;
            int v = edges[e].v - 1;

            t[e] = model->addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS);

            c1[e] = model->addConstr(t[e] >= x[u] - x[v]);
            c2[e] = model->addConstr(t[e] >= -x[u] + x[v]);

            if(edges[e].directed)
            {
                model->addConstr(x[v] - x[u] == edges[e].weight); //here we are saying that directed edges are hard constraints
                //I do this because the only time we have directed edges is from the reduction from feas-BLP so we know it will work
            }
        }

        model->set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    }
    catch (GRBException& e)
    {
        std::cerr << "constructor error " 
                  << e.getErrorCode() << " - "
                  << e.getMessage() << "\n";
        throw;  // rethrow so caller sees the error
    }
}

std::pair<double, std::vector<double>> ExactEmbeddingSolver::solve(
    const std::vector<int>& fixed_signs
)
{
    try
    {
        const int m = edges.size();

        // ---------------- Update RHS ----------------
        for (int e = 0; e < m; ++e)
        {
            double b = fixed_signs[e] * edges[e].weight;

            // Constraint 1:
            // t_e - x_u + x_v >= -b
            c1[e].set(GRB_DoubleAttr_RHS, -b);

            // Constraint 2:
            // t_e + x_u - x_v >=  b
            c2[e].set(GRB_DoubleAttr_RHS,  b);
        }

        // IMPORTANT: ensure Gurobi sees RHS updates
        model->update();

        // ---------------- Optimize ----------------
        model->optimize();

        int status = model->get(GRB_IntAttr_Status);

        if (status == GRB_OPTIMAL)
        {
            double obj = model->get(GRB_DoubleAttr_ObjVal);

            std::vector<double> embedding(x.size());
            for (int i = 0; i < static_cast<int>(x.size()); ++i)
            {
                embedding[i] = x[i].get(GRB_DoubleAttr_X);
            }

            return {obj, embedding};
        }

        // Infeasible or numerical issue → treat as infinite
        if (status == GRB_INFEASIBLE ||
            status == GRB_INF_OR_UNBD ||
            status == GRB_UNBOUNDED)
        {
            return {std::numeric_limits<double>::infinity(), {}};
        }

        // Any other status → safe fallback
        return {std::numeric_limits<double>::infinity(), {}};
    }
    catch (GRBException& e)
    {
        std::cerr << "Gurobi exception in solve(): "
                  << e.getMessage() << "\n";
        return {std::numeric_limits<double>::infinity(), {}};
    }
}


int select_branching_edge(
    const std::vector<int>& fixed_signs
){
    for(int i=0; i<static_cast<int>(fixed_signs.size()); ++i){
        if(fixed_signs[i] == 0){
            return i;
        }
    }

    return -1;
}

std::vector<std::vector<int>>
connected_dfs_ordering(
    const std::vector<std::vector<Adjacency>>& adj_list
){
    std::vector<std::vector<int>> components;
    std::unordered_set<int> visited;

    for (int start_vertex=1; start_vertex < static_cast<int>(adj_list.size())+1; ++start_vertex)
    {
        if (visited.count(start_vertex))
            continue;

        std::vector<int> component;
        std::stack<int> st;

        st.push(start_vertex);
        visited.insert(start_vertex);

        while (!st.empty())
        {
            int v = st.top();
            st.pop();
            component.push_back(v);

            // Traverse ALL neighbors regardless of direction
            for (const Adjacency& adj : adj_list[v-1])
            {
                int u = adj.neighbourId;

                if (!visited.count(u))
                {
                    visited.insert(u);
                    st.push(u);
                }
            }
        }

        components.push_back(component);
    }

    return components;
}


std::pair<bool, std::vector<double>> obtain_feasibility_from_ordering(
    const std::vector<std::vector<Adjacency>>& adj_list,
    const std::vector<double>& t_best
){
    //this function will determine if the ordering of the vertices induced by t_best allows for a feasible embedding of the vertices on the 1D line
    
    int n {static_cast<int>(t_best.size())};

    //recall that t_best[i] is the position of the vertex with index i+1, that is how we can look in adj_list

    std::vector<double> final_embedding(n, 0); //this is the vector that represent the embedding we are constructing
    std::vector<std::vector<int>> dfs_ordering = connected_dfs_ordering(adj_list); // functions returns a vector of vectors of int, each vector represents a connected component
    //the ordering of indices inside each vector represents the DFS ordering of the vertices 
    std::unordered_set<int> placed {};
    constexpr double EPS = 1e-8;


    //iterating over each connected component of the graph
    for(const auto& cc: dfs_ordering){
        //iterating over each vertex in the connected component
        placed.insert(cc[0]); //position the first vertex in the ordering for each connected component
        final_embedding[cc[0]-1] = 0.0;
        for(int i=1; i<static_cast<int>(cc.size()); ++i){
            int v_id = cc[i];
            const std::vector<Adjacency>& ngbrs = adj_list[v_id-1];
            if(ngbrs.size() == 0){
                final_embedding[v_id-1] = 0.0; //since the vertex has no neighbours we can place it anywhere, we thus place it 0.0 for convenience
                placed.insert(v_id);
            }else{
                double cand_pos = 0.0;
                bool has_candidate = false;

                for (const auto& adj : ngbrs)
                {
                    int u = adj.neighbourId;

                    if (placed.count(u))
                    {
                        double expected;

                        if(adj.directed){
                            if(adj.outgoing)
                                expected = final_embedding[u-1] + adj.dist;
                            else
                                expected = final_embedding[u-1] - adj.dist;
                        }else{
                            if (t_best[v_id-1] <= t_best[u-1])
                                expected = final_embedding[u-1] - adj.dist;
                            else
                                expected = final_embedding[u-1] + adj.dist;
                        }

                        if (!has_candidate)
                        {
                            cand_pos = expected;
                            has_candidate = true;
                        }
                        else
                        {
                            if (std::abs(cand_pos - expected) > EPS)
                                return {false, {}};
                        }
                    }
                }

                if (!has_candidate)
                {
                    // No placed neighbors — shouldn't happen in DFS
                    return {false, {}};
                }

                final_embedding[v_id-1] = cand_pos;
                placed.insert(v_id);

            }


        }
    }

    return {true, final_embedding};


}

void branch_and_bound(
    ExactEmbeddingSolver& exact_solver,
    const std::vector<CycleID>& i_cycles,
    const std::vector<IndexedEdge>& i_edges,
    std::vector<int>& fixed_signs,
    double& bestUB,
    std::vector<double>& bestEmbedding,
    int scale
){

    if(bestUB == 0.0){
        return;
    }

    double LB = compute_cycle_packing_LB(i_cycles, fixed_signs, i_edges, scale);
    if(LB >= bestUB){
        //std::cerr << "pruned LB >= UB, " << LB << " " << bestUB << "\n";
        return;
    }
    int eid = select_branching_edge(fixed_signs);



    if(eid == -1){
    //have no more nodes to select, we have decided on all of them
    auto solver_pair = exact_solver.solve(fixed_signs);

    if(solver_pair.first < bestUB){
        bestUB = solver_pair.first;
        bestEmbedding = solver_pair.second;
        if(solver_pair.first == 0.0){
            std::cerr << "found embedding with error 0 with signed s \n";
            return;
        }
    }
    return;
}


    //branch 1
    fixed_signs[eid] = 1;
    branch_and_bound(exact_solver, i_cycles, i_edges, fixed_signs, bestUB, bestEmbedding, scale);
    //branch2
    fixed_signs[eid] = -1;
    branch_and_bound(exact_solver, i_cycles, i_edges, fixed_signs, bestUB, bestEmbedding, scale);

    fixed_signs[eid] = 0;
}

SolverConfig getSolverConfig(int weight_flag){
    SolverConfig sc;
    if(weight_flag == 1){
        sc.mode = WeightMode::INTEGER;
        sc.scaling_power = 0;
    }else{
        sc.mode = WeightMode::REAL_SCALED;
        sc.scaling_power = 1; //only care about 2 decimals
    }
    return sc;
}

std::vector<std::vector<int>> get_edge_to_cycles(std::vector<CycleID> i_cycles, int num_edges){
    std::vector<std::vector<int>> edges_to_cycles(num_edges);
    for(int c=0; c < static_cast<int>(i_cycles.size()); ++c){
        for(int e_id: i_cycles[c].edge_ids){
            edges_to_cycles[e_id].push_back(c);
        }
    }
    return edges_to_cycles;
}

std::pair<double, std::vector<double>> solve_minerr_dgp1(
    const std::string& filename,
    int weight_flag
){
    double epsThreshold = 2.0;

    SolverConfig solverconfig = getSolverConfig(weight_flag);

    
    auto [edges, vertex_ids, vertex_id_to_name] = parse_dgp_instance_dat_file(filename); // we assume that the set of vertex ids corresponds to [|n|] = {1,...,n}
    

    std::vector<std::vector<Adjacency>> adj_list = create_adjacency_list_from_edges(edges, vertex_ids.size());
    std::vector<std::vector<Edge>> spanning_tree_edges = get_spanning_forest(adj_list);
    std::vector<std::vector<std::vector<Edge>>> cycle_basis = get_cycle_basis(spanning_tree_edges, adj_list);

    int scale = static_cast<int>(std::pow(10, solverconfig.scaling_power));
    std::vector<IndexedEdge> i_edges = build_indexed_edges(edges, scale); //i_edges[id] has .id = id
    std::vector<CycleID> i_cycles =  convert_cycles_to_edge_ids(cycle_basis, i_edges);

    
    std::vector<int> fixed_signs = std::vector<int>(static_cast<int>(edges.size()), 0); //initalize it so that all edges are free edges

    //here we encoding the directed edges, which are not free at all
    for(int i=0; i<static_cast<int>(edges.size()); ++i){
        if(edges[i].directed == 1){
            fixed_signs[i] = -1;
        }
    }
    std::pair<double, std::vector<double>> p_best = optimized_projection_minErrDGP1_UB(edges, vertex_ids, adj_list); //obtain a tight UB with our heuristic
    double bestUB = p_best.first;

    double initialLB = compute_cycle_packing_LB(i_cycles, fixed_signs, i_edges, scale);
    std::cerr << "Initial UB " << bestUB << "\n";
    std::cerr << "intial LB " << initialLB << "\n";

    std::vector<double> t_best = p_best.second;
    if(bestUB <= epsThreshold){
        //in this case we try to obtain the feasibility of the embedding from the ordering provided by the heuristic
        std::pair<bool, std::vector<double>> p_order = obtain_feasibility_from_ordering(adj_list, t_best);
        if(p_order.first){
            //if the instance is feasible we simply return 0 as that will be the minimum error
            return {0, p_order.second};
        }

    }



    ExactEmbeddingSolver solver(i_edges, vertex_ids.size());
    std::vector<double> bestEmbedding(vertex_ids.size(), 0.0);
    branch_and_bound(solver, i_cycles, i_edges, fixed_signs, bestUB, bestEmbedding, scale);

    return {bestUB, bestEmbedding};
}