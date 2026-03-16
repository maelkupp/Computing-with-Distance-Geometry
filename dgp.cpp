#include "dgp.h"
#include "random.h"
#include <tuple>
#include <vector>
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

constexpr double PI = 3.14159265358979323846;


std::tuple<std::vector<Edge>, std::set<int>, std::unordered_map<int, std::string>> parse_dgp_instance_dat_file(std::string dgp_dat_file_path){
    std::vector<Edge> edges;
    std::set<int> vertex_ids;
    std::unordered_map<int, std::string> vertex_id_to_name;

    std::ifstream infile(dgp_dat_file_path);
    if (!infile) {
        std::cerr << "Error opening file: " << dgp_dat_file_path << "\n";
        return {edges, vertex_ids, vertex_id_to_name};
    }

    std::string line;
    bool in_edge_block = false;

    while (std::getline(infile, line)) {

        // trim leading whitespace
        auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        line = line.substr(first);

        // skip full-line comments
        if (line.starts_with("#")) continue;

        // start of edge block
        if (line.starts_with("param : E")) {
            in_edge_block = true;
            continue;
        }

        // end of edge block
        if (in_edge_block && line == ";") {
            in_edge_block = false;
            continue;
        }

        if (!in_edge_block) {
            // ignore param lines for now
            continue;
        }

        // ----- EDGE LINE -----

        // split comment
        std::string data_part = line;
        std::string comment_part;

        size_t hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            data_part = line.substr(0, hash_pos);
            comment_part = line.substr(hash_pos + 1);
        }

        // parse numeric part
        std::stringstream ss(data_part);

        int u, v, flag;
        double dist;

        if (!(ss >> u >> v >> dist >> flag)) {
            std::cerr << "Malformed edge line: " << line << "\n";
            continue;
        }

        edges.emplace_back(u, v, dist, flag);
        vertex_ids.insert(u);
        vertex_ids.insert(v);

        // parse names from comment if present
        if (!comment_part.empty()) {

            size_t lbracket = comment_part.find('[');
            size_t rbracket = comment_part.find(']');

            if (lbracket != std::string::npos && rbracket != std::string::npos) {

                std::string names =
                    comment_part.substr(lbracket + 1,
                                         rbracket - lbracket - 1);

                size_t comma = names.find(',');
                if (comma != std::string::npos) {

                    std::string u_name = names.substr(0, comma);
                    std::string v_name = names.substr(comma + 1);

                    if (!vertex_id_to_name.count(u))
                        vertex_id_to_name[u] = u_name;
                    if (!vertex_id_to_name.count(v))
                        vertex_id_to_name[v] = v_name;
                }
            }
        }
    }
    return {edges, vertex_ids, vertex_id_to_name};
}

void write_edges_to_dat_file(const std::vector<Edge>& edges) {
    std::ofstream file("temp_graph.dat");
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file temp_graph.dat");
    }

    // ---- compute number of vertices n ----
    int n = 0;
    for (const auto& e : edges) {
        n = std::max(n, std::max(e.u, e.v));
    }

    // ---- timestamp ----
    std::time_t t = std::time(nullptr);
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%y%m%d%H%M%S", std::localtime(&t));

    // ---- header ----
    file << "# DGP instance in .dat format for dgp.mod\n";
    file << "# written by C++ exporter on " << time_buffer << "\n";
    file << "# this graph is a DGP encoding of random instance\n\n";

    file << "param n := " << n << ";\n";
    file << "param nlits := 0;\n";
    file << "param Kdim := 1;\n";
    file << "param : E : c I :=\n";

    // ---- edges ----
    file << std::fixed << std::setprecision(3);
    for (const auto& e : edges) {
        file << "  "
             << e.u << " "
             << e.v << " "
             << e.weight << "  "
             << (e.directed ? 1 : 0)
             << "  # [" << "x_"<< e.u << ", " << "x_" << e.v << "]"
             << "\n";
    }

    // ---- footer ----
    file << ";\n";

    file.close();
}

//function that looks at the adjacency list of an DGP instance and returns a vector of sets corresponding to the vertex ids in the distinct connected component of the DGP instance
std::vector<std::set<int>> get_connected_components(
    std::unordered_map<int, std::vector<Adjacency>>& adj_list,
    const std::set<int>& vertex_ids){

        std::vector<std::set<int>> components;
        std::set<int> visited;

        for(int id: vertex_ids){
            auto it = visited.find(id);
            if(it == visited.end()){
                //the id has not yet been seen, we start a new traversal from it
                std::set<int> component = dfs_through_adj_list(adj_list, id);
                components.push_back(component);
                for(int visited_id: component){
                    visited.insert(visited_id);
                }
            }
        }
        return components;
}

std::unordered_map<int, std::vector<Adjacency>> create_adjacency_list_from_edges(const std::vector<Edge>& edges, const int num_vertices){
    //function that creates an adjacency list of a DGP instance from a vector of Edges
    std::cerr << "Creating an adjacency list for a DGP instance with " << edges.size() << "edges \n";
    std::unordered_map<int, std::vector<Adjacency>> adj_list;

    for(int i=1; i<=num_vertices; ++i){
        //initialise each adjacency list of the vertices to the empty list
        adj_list[i] = {};
    }

    for(const Edge& edge: edges){
        if(edge.directed){
            adj_list[edge.u].push_back(Adjacency(edge.v, edge.weight, true));
            adj_list[edge.v].push_back(Adjacency(edge.u, edge.weight, false));
        }else{
            adj_list[edge.u].push_back(Adjacency(edge.v, edge.weight));
            adj_list[edge.v].push_back(Adjacency(edge.u, edge.weight));    
        }
    }

    return adj_list;
}


std::pair<std::vector<Edge>, std::unordered_set<int>> dfs_for_spanning_tree(std::unordered_map<int, std::vector<Adjacency>>& adj_list, int start_vertex){
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
            // find distance from parent -> u
            tree_edges.push_back({parent, u, dist, 0});

        }

        for (const Adjacency& adj : adj_list[u]) {
            int v = adj.neighbourId;
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                stk.push({u, v, adj.dist});
            }
        }
    }

    return {tree_edges, visited};
}

std::vector<std::vector<Edge>> get_spanning_forest(std::unordered_map<int, std::vector<Adjacency>>& adj_list){
    //this function returns a vector of edges (all assumed to be undirected) corresponding to a spanning tree of a graph represented with its
    //adjacency list, this is done by performing a DFS through the tree
    //if there are multiple connected components in our graph this will return a spanning forest
    
    std::vector<std::vector<Edge>> spanning_forest {};
    std::unordered_set<int> visited {};

    for(const auto& [v_id, list]: adj_list){
        if(visited.find(v_id) == visited.end()){
            //we have not visited the ID yet
            auto tree_pair = dfs_for_spanning_tree(adj_list, v_id);
            std::vector<Edge> spanning_tree = tree_pair.first;
            std::unordered_set<int> visited_tree = tree_pair.second;

            spanning_forest.push_back(spanning_tree);
            for(const auto& visited_id: visited_tree){
                visited.insert(visited_id);
            }
        }
    }
    return spanning_forest;
}

std::pair<
    std::unordered_map<int, Edge>,
    std::unordered_map<int, int>
>
get_parent_and_depth_maps_DFS(
    std::unordered_map<int, std::vector<Adjacency>>& adj_list,
    int root_id
){
    // parent_edge[v] = edge connecting v to its parent in the DFS tree
    std::unordered_map<int, Edge> parent_edge;
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

        for (const auto& adj : adj_list[u]) {
            int v = adj.neighbourId;

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

std::unordered_set<std::pair<int, int>, PairHash> create_tree_edge_set(const std::vector<Edge>& spanning_tree_edges){
    std::unordered_set<std::pair<int, int>, PairHash> tree_edges_set{};
    for(const auto& edge: spanning_tree_edges){
        tree_edges_set.insert({std::min(edge.u, edge.v), std::max(edge.u, edge.v)});
    }
    return tree_edges_set;
}

// Helper: find one simple cycle via DFS
bool dfs_find_cycle(
    int u,
    int parent,
    std::unordered_map<int, std::vector<Adjacency>>& adj,
    std::unordered_map<int, bool>& visited,
    std::unordered_map<int, int>& parent_map,
    std::vector<int>& cycle_nodes
) {
    visited[u] = true;
    for (auto& adjInfo : adj[u]) {
        int v = adjInfo.neighbourId;
        // Skip the parent edge
        if (v == parent) continue;
        if (!visited[v]) {
            parent_map[v] = u;
            if (dfs_find_cycle(v, u, adj, visited, parent_map, cycle_nodes)) {
                return true;
            }
        } else {
            // found a back edge -> cycle
            // reconstruct cycle from u back to v
            int cur = u;
            cycle_nodes.push_back(v);
            while (cur != v) {
                cycle_nodes.push_back(cur);
                cur = parent_map[cur];
            }
            cycle_nodes.push_back(v); // close cycle
            return true; 
        }
    }
    return false;
}

std::vector<std::vector<std::vector<Edge>>> get_greedy_edge_disjoint_cycles(
    std::vector<std::vector<Edge>>& spanning_forest_edges,
    std::unordered_map<int, std::vector<Adjacency>>& adj_list
) {
    // result per component
    std::vector<std::vector<std::vector<Edge>>> all_disjoint_cycles;

    // We will keep a working adjacency list we can modify
    auto working_adj = adj_list;

    // For each component's tree (so we preserve spanning forest structure)
    for (const auto& tree : spanning_forest_edges) {
        // We grab only the edges that remain in the working adjacency list
        std::vector<std::vector<Edge>> comp_cycles;
        
        while (true) {
            // We need to find any cycle in the current graph
            std::unordered_map<int, bool> visited;
            std::unordered_map<int, int> parent_map;
            std::vector<int> cycle_nodes;

            bool found_cycle = false;
            // Try DFS from every node that still has neighbors
            for (auto &kv : working_adj) {
                int start = kv.first;
                if (!visited[start] && !working_adj[start].empty()) {
                    parent_map[start] = -1;
                    if (dfs_find_cycle(start, -1, working_adj, visited, parent_map, cycle_nodes)) {
                        found_cycle = true;
                        break;
                    }
                }
            }

            if (!found_cycle) break; // no more cycles

            // cycle_nodes now holds the cycle vertex sequence (start == end)
            // We convert consecutive vertices into edges
            std::vector<Edge> cycle_edges;
            for (size_t i = 0; i + 1 < cycle_nodes.size(); ++i) {
                int a = cycle_nodes[i];
                int b = cycle_nodes[i + 1];

                // find adjacency to get weight
                auto &neighbors = working_adj[a];
                double w = 0;
                for (auto &adj : neighbors) {
                    if (adj.neighbourId == b) {
                        w = adj.dist;
                        break;
                    }
                }
                cycle_edges.push_back(Edge(a, b, w, 0));
            }

            // Add this cycle to the component
            comp_cycles.push_back(cycle_edges);

            // Remove edges of this cycle from the working adjacency list
            for (const auto &e : cycle_edges) {
                int u = e.u;
                int v = e.v;
                // remove from both adjacency lists
                auto &listU = working_adj[u];
                listU.erase(std::remove_if(listU.begin(), listU.end(),
                    [&](const Adjacency &adj){ return adj.neighbourId == v; }),
                    listU.end());
                auto &listV = working_adj[v];
                listV.erase(std::remove_if(listV.begin(), listV.end(),
                    [&](const Adjacency &adj){ return adj.neighbourId == u; }),
                    listV.end());
            }
        }

        if (!comp_cycles.empty()) {
            all_disjoint_cycles.push_back(comp_cycles);
        }
    }

    return all_disjoint_cycles;
}

std::vector<std::vector<std::vector<Edge>>> get_cycle_basis(std::vector<std::vector<Edge>>& spanning_forest_edges, std::unordered_map<int, std::vector<Adjacency>>& adj_list){
    //this function will take a spanning tree and the adjacency list of the entire graph and will return a cycle basis of the entire graph
   std::vector<std::vector<std::vector<Edge>>> cycle_basis;
   //adjacency list representing only the spanning tree


   for(const auto& spanning_tree_edges: spanning_forest_edges){
    //preprocessing for each tree in the forest
    std::unordered_set<std::pair<int, int>, PairHash> tree_edge_set = create_tree_edge_set(spanning_tree_edges);
    if(spanning_tree_edges.size() == 0) continue;
    std::unordered_map<int, std::vector<Adjacency>> tree_adj_list = create_adjacency_list_from_edges(spanning_tree_edges, spanning_tree_edges.size());
    auto [parent_edge_tree, depth_tree] = get_parent_and_depth_maps_DFS(tree_adj_list, spanning_tree_edges[0].u); // get an arbitrary vertex id that is in the tree from the first element in the vector
    std::vector<std::vector<Edge>> component_cycle_basis;


    for(auto& it: adj_list){
        //iterating over all edges of the graph
        for(auto& adj: adj_list[it.first]){
            //iterating over each neighbour
            
            //saving steps and duplicates
            if(adj.neighbourId >= it.first) continue;
                
            // the edge is not in this connected component
            if(parent_edge_tree.find(it.first) == parent_edge_tree.end() || parent_edge_tree.find(adj.neighbourId) == parent_edge_tree.end()) continue;             
            
            //the edge is a tree edge so we skip it
            if(tree_edge_set.find({it.first, adj.neighbourId}) != tree_edge_set.end()) continue;
                
            //now dealing with a non-tree edge

            std::vector<Edge> cycle {}; // initialise an empty cycle
                int x = it.first;
                int y = adj.neighbourId;

                // Lift deeper vertex until depths match
                while (depth_tree[x] > depth_tree[y]) {
                    Edge e = parent_edge_tree[x];
                    cycle.push_back(e);
                    x = (e.u == x ? e.v : e.u);
                }

                while (depth_tree[y] > depth_tree[x]) {
                    Edge e = parent_edge_tree[y];
                    cycle.push_back(e);
                    y = (e.u == y ? e.v : e.u);
                }

                // Lift both simultaneously until lowest common ancestor is reached
                while (x != y) {
                    Edge ex = parent_edge_tree[x];
                    Edge ey = parent_edge_tree[y];
                    cycle.push_back(ex);
                    cycle.push_back(ey);
                    x = (ex.u == x ? ex.v : ex.u);
                    y = (ey.u == y ? ey.v : ey.u);
                }

                // ---- 4. Close the cycle with the non-tree edge ----
                cycle.push_back(Edge(it.first, adj.neighbourId, adj.dist, 0));

                component_cycle_basis.push_back(cycle);
        }
    }
    if(component_cycle_basis.size() > 0){
        cycle_basis.push_back(component_cycle_basis);

    }

   }
   return cycle_basis;
}


// ------------ computing the MinErrDGP of the cycle basis using DP ------------------
double DP_cycle_error_real(std::vector<double> cycle) {
    int n = cycle.size();
    if (n == 0) return 0.0;

    // compute total sum
    double S = std::accumulate(cycle.begin(), cycle.end(), 0.0);

    // resolution controls precision of DP
    const double resolution = 1e-4;

    // max possible sum scaled
    int maxSum = static_cast<int>(std::floor(S / resolution + 0.5));

    // dp array — reachable subset sums
    std::vector<char> dp(maxSum + 1, 0);
    dp[0] = 1;

    for (double w : cycle) {
        int step = static_cast<int>(std::floor(w / resolution + 0.5));
        for (int j = maxSum - step; j >= 0; --j) {
            if (dp[j]) dp[j + step] = 1;
        }
    }

    // Find p closest to S/2
    int target = static_cast<int>(std::floor((S / 2) / resolution + 0.5));
    int best = 0;

    for (int p = 0; p <= maxSum; ++p) {
        if (!dp[p]) continue;
        if (std::abs(p - target) < std::abs(best - target)) {
            best = p;
        }
    }

    // compute resulting error
    double p_val = best * resolution;
    double cycle_error = std::abs(2 * p_val - S);
    std::cerr << "Got additional error from cycle " << cycle_error << "\n";
    return cycle_error;
}


double DP_cycle_error(const std::vector<double>& cycle) {
    int n = cycle.size();
    if (n == 0) return 0.0;

    int S = std::accumulate(cycle.begin(), cycle.end(), 0);

    std::vector<char> dp(S + 1, 0);
    dp[0] = 1;

    for (int w : cycle) {
        for (int j = S - w; j >= 0; --j) {
            if (dp[j]) dp[j + w] = 1;
        }
    }

    int target = S / 2;
    int best = 0;

    for (int p = 0; p <= S; ++p) {
        if (!dp[p]) continue;
        if (std::abs(p - target) < std::abs(best - target)) {
            best = p;
        }
    }

    double cycle_error {std::abs(2 * best - S)};
    std::cerr << "Got additional error on a cycle " << cycle_error << "\n";
    return cycle_error;
}

double compute_minErrDGP_cycle(std::vector<std::vector<std::vector<Edge>>>& cycle_basis, bool real_edge_weights){
    //function takes a cycle basis of our graph and computes the minimum error of this basis, acting as a lower bound for the true minimum error
    double tot_err {0.0};

    for(const auto& connected_component_cycle: cycle_basis){
        for(const auto& e_disjoint_cycle: connected_component_cycle){
            std::vector<double> cycle_values {};
            for(const Edge& e: e_disjoint_cycle){
                cycle_values.push_back(e.weight);
            }
            
            if (real_edge_weights){
                tot_err += DP_cycle_error_real(cycle_values);
            }else{
                tot_err += DP_cycle_error(cycle_values);
            }
        }
    }

    return tot_err;
}

//function that displays a 1 dimension embedding of a DGP instance
void display_1Dembedding(std::unordered_map<int, double> embedding, std::unordered_map<int, std::string> vertex_id_2_name){
    for(const auto& [vertex_id, pos]: embedding){
        std::cerr << "Placed vertex " << vertex_id << " " << vertex_id_2_name[vertex_id] << " at " << pos << "\n";
    }
}


std::set<int> dfs_through_adj_list(std::unordered_map<int, std::vector<Adjacency>>& adj_list, int id){
    /*this function performs a depth first search through the DGP instance being going through the adjacency list starting from the id past as a parameter
    it then returns an unordered_set of all the ids of this connnected component
    */
    //initiate the set of components for this connected component with the id (the root of our DFS)
    std::set<int> component_vertices {id};
    std::stack<int> stk;
    for(auto& adjacency: adj_list[id]){
        stk.push(adjacency.neighbourId);
        component_vertices.insert(adjacency.neighbourId);
    }
    while(stk.size() > 0){
        int neighbour_id = stk.top();
        stk.pop();
        for(auto& adjacency: adj_list[neighbour_id]){
            auto it = component_vertices.find(adjacency.neighbourId);
            if(it == component_vertices.end()){
                //the id has not already been traversed so we push it to the stack so that it can be popped and traversed
                stk.push(adjacency.neighbourId);
                component_vertices.insert(adjacency.neighbourId);
            }
        }

    }
    return component_vertices;

}


double compute_vertex_embedding_error(int vertex_id_to_place, std::unordered_map<int, double> current_embedding,
    std::unordered_map<int, std::vector<Adjacency>> adj_list, std::unordered_map<int, int>& pos){
    //compute the additional error incurred from placing the vertex_id_to_place at a position

    //to not count the same error twice we only add the error if the added vertex is earlier in the ordering
    double delta_error {0.0};
    double vertex_position = current_embedding[vertex_id_to_place];
    for(const auto& adj: adj_list[vertex_id_to_place]){
        if(current_embedding.find(adj.neighbourId) != current_embedding.end()){
            int neighbour_index = pos[adj.neighbourId];
            int vertex_index = pos[vertex_id_to_place];
            if(vertex_index > neighbour_index){
                delta_error += std::abs( std::abs(vertex_position - current_embedding.at(adj.neighbourId)) -  adj.dist);
            }
        }
    }
    return delta_error;
}

double verbose_compute_1Dembedding_error(std::vector<Edge> edges, std::unordered_map<int, double> embedding, std::unordered_map<int, std::string> vertex_id_to_name){
    //this function takes a DGP1 instance under the form of a vector of edges and an embedding of this instance and computes the minErrDGP1 in a verbose way
    std::cerr << "\n";
    double total_error {0.0};
    for(const auto& edge: edges){
        double error = std::abs(std::abs(embedding[edge.u] - embedding[edge.v]) - edge.weight);
        if(error > 0){
            //the embedding of this edge is causing an error
            std::cerr << "Edge " << vertex_id_to_name[edge.u] << " - " << vertex_id_to_name[edge.v] << " with weight " << edge.weight << " embedded at " << embedding[edge.u] << " " << embedding[edge.v] << "\n";
            std::cerr << "Adding error " << error << " to previous total error " << total_error << " to make total error " << total_error + error << "\n";
            total_error += error;
        }
    }
    return total_error;
}


// ---------------------------- things to do with 2D instances and generating them ---------------


Point random_point(double& A){
    return Point(Random::get(0.0, A),Random::get(0.0, A));
}



std::vector<Point> generate_points(int n, double A){
    std::vector<Point> points;
    //.reserve to avoid multiple allocations as we know the size in advance
    points.reserve(n);
    //have a random generator
    std::mt19937 gen(41);
    for(int i=0; i<n; ++i){
        points.push_back(random_point(A));
    }

    return points;

}

std::vector<Edge> generate_edges_across_points(std::vector<Point>& points, double p){
    //returns a random vector of edges, each edge in the graph is present with a probability p
    std::mt19937 gen(10); // fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<Edge> edges;
    int l = points.size();
    // important the vertex indices start at 1 and not at 0
    for(int i=1; i<=l; ++i){
        //start at i+1 to not have duplicate edges or self edges
        for(int j=i+1; j<=l; ++j){
            double r = dist(gen);
            if(r<p){
                //dont want simple graph as they are redundant
                //note the difference between the vertex ids and the index of the points in the vector, have to decrement by one
                edges.push_back(Edge(i, j, euclidean_distance(points[i-1], points[j-1]), false));
            }

        }
    }

    return edges;

}


std::pair<std::vector<Edge>, std::vector<Point>> generate_2d_dgp_instance(int n, double p, double A){
    std::cerr << "Generating " << n << " points in [0," << A << "]x[0," << A << "] with edge density " << p << std::endl;

    std::vector<Point> points = generate_points(n, A);
    std::cerr << "Have sampled" << n << "points in [0," << A << "] x [0,"<< A << "]" << std::endl;
    for(auto& point: points){
        std::cerr << point.x << "," <<  point.y << "\t";
    }
    //vertex list is just a list of ids which will go from 0 -> n-1 for us, it is implicit in our definition of points
    //we are using that vertex with vertex id i is embedded at points[i]

    //we represent edges as a tuple that contains (vertex_id1, vertex_id2, edge weight or distance, int=0 or 1 to indicate if it is directed or not)
    std::vector<Edge> edges = generate_edges_across_points(points, p);
    return {edges, points};
}

double A_l(const Edge& edge, const Line& line, const std::vector<Point>& points){
    //returns <edge.u - edge.v, line.vec>
    Point u = points[edge.u-1];
    Point v = points[edge.v-1];
    return (u.x - v.x)*line.dir.x + (u.y - v.y)*line.dir.y;
}

double B_l(const Edge& edge, const Line& line, const std::vector<Point>& points){
    //return det(edge.u - edge.v, line.vec)
    Point u = points[edge.u-1];
    Point v = points[edge.v-1];
    return (u.x - v.x)*line.dir.y - (u.y - v.y)*line.dir.x;
}

double optimal_rotation(const std::vector<Edge>& edges, const Line& line, const std::vector<Point>& points){
    //this function uses the closed form equation I derived to find the angle to rotate the points of the embedded DGP
    // to best align the edges with a line, making them as parallel as possible to this line
    // \theta^\star = \frac{1}{2}atan2(\sum_{\ell \in E}A_\ell B_\ell, \frac{1}{2}\sum_{\ell \in E}(A_\ell^2 - B_\ell^2))
    // A_\ell = \langle x_i - x_j, u\rangle B_\ell = \det(x_i - x_j, u)
    // where u is the line we want to the line (copy paste this in overleaf to see what it looks like)
    double crossSum {0.0};
    double diffSquares {0.0};
    for(const auto& edge: edges){
        double a = A_l(edge, line, points);
        double b = B_l(edge, line, points);
        crossSum += a*b;
        diffSquares += (a*a - b*b);
    }

    //divide beta by two to match the formula
    diffSquares /= 2;

    return atan2(crossSum, diffSquares)/2;
}

// ----------- functions for obtaining a cheap upper bound to the MinErrDGP1 using an 2D embedding of the instance

double cheap_rotation_minErrDGP1_UB(const std::vector<Edge>& edges, const std::vector<Point>& points){
    std::cerr << "Computing the cheap rotation based upper bound for the minErrDGP1 \n";

    //perform the rotation around the x_axis for simplicity
    double theta = optimal_rotation(edges, Line(Point(0,0), 1, 0), points);


    //will rotate the x_axis -theta counterclockwise (or theta clockwise) same as rotating each point by theta counterclockwise

    double c = cos(-theta);
    double s = sin(-theta);

    double x_dir_new = c;
    double y_dir_new = s;
    Line optimal_projection_line = Line(Point(0,0), x_dir_new, y_dir_new);

    double UB_error {0.0};
    for(auto& [i1, i2, weight, dir]: edges){
        double proj_err = optimal_projection_line.projection_error(points[i1-1], points[i2-1]);
        UB_error += proj_err;
    }

    std::cerr << "UB error is " << UB_error << "\n";
    return UB_error;
}


double local_obj_abs(
    double x,
    int i_vertex_id_1based,
    const std::unordered_map<int, std::vector<Adjacency>>& adj_list,
    const std::vector<double>& t
){
    double sum = 0.0;
    const auto& nbrs = adj_list.at(i_vertex_id_1based);
    for (const auto& nbr : nbrs) {
        int j = nbr.neighbourId - 1;   // 0-based index into t
        sum += std::abs(std::abs(x - t[j]) - nbr.dist);
    }
    return sum;
}

double coefficient_descent_on_line(const std::vector<Edge>& edges,
    const std::unordered_map<int, std::vector<Adjacency>>& adj_list,
    std::vector<double>& t){
    auto global_error = [&](const std::vector<double>& tt){
        double E = 0.0;
        for (const auto& e : edges) {
            E += std::abs(std::abs(tt[e.u - 1] - tt[e.v - 1]) - e.weight);
        }
        return E;
    };

    double prev_E = global_error(t);
    std::cerr << "Initial 1D error: " << prev_E << "\n";

    const double lambda = 0.5;
    const double rel_tol = 5e-5;
    const int max_sweeps = 100;


    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        // Jacobi-style buffer for stability
        std::vector<double> t_new = t;

        for (int i = 0; i < (int)t.size(); ++i) {
            int vid = i + 1;
            auto it = adj_list.find(vid);
            if (it == adj_list.end() || it->second.empty()) continue;

            // Build candidate set: {t_j ± d_ij} plus current t_i
            std::vector<double> cand;
            cand.reserve(2 * it->second.size() + 1);
            cand.push_back(t[i]);

            for (const auto& nbr : it->second) {
                int j = nbr.neighbourId - 1;
                double dij = nbr.dist;
                cand.push_back(t[j] + dij);
                cand.push_back(t[j] - dij);
            }

            // Pick best candidate for the TRUE local objective
            double best_x = cand[0];
            double best_val = local_obj_abs(best_x, vid, adj_list, t);
            for (int k = 1; k < (int)cand.size(); ++k) {
                double val = local_obj_abs(cand[k], vid, adj_list, t);
                if (val < best_val) {
                    best_val = val;
                    best_x = cand[k];
                }
            }

            // Damped update
            t_new[i] = (1.0 - lambda) * t[i] + lambda * best_x; //move part of the way towards the locally optimal position
            //if we jump all the way we might have strong reactions to conflicting constraints etc
        }

        t.swap(t_new); // same as t = t_new but faster

        double curr_E = global_error(t);
        double rel_impr = std::abs(prev_E - curr_E) / std::max(1.0, prev_E);

        if (rel_impr < rel_tol) {
            std::cerr << "Converged after " << sweep + 1 << " sweeps.\n";
            break;
        }

        prev_E = curr_E;


        std::cerr << "After sweep: " << curr_E << "\n";
    }

    double Efinal = global_error(t);
    std::cerr << "Optimized error: " << Efinal << "\n";
    return Efinal;
}


static inline double wrap_angle_pi(double theta) {
    // For line directions in 1D projection, angles are equivalent mod pi
    // because u and -u define the same line. Keep theta in [0, pi).
    double pi = PI;
    theta = std::fmod(theta, pi);
    if (theta < 0) theta += pi;
    return theta;
}

double optimized_projection_minErrDGP1_UB(
    const std::vector<Edge>& edges,
    const std::vector<Point>& points
){
    std::cerr << "Computing rotation + 1D coordinate descent UB (multi-angle around theta*)\n";

  
    const double theta_init {0.0};

    // Build adjacency once (independent of angle)
    auto adj_list = create_adjacency_list_from_edges(edges, edges.size());

    // Small deviations around theta* (in radians). can tweak these variations
    // Here: 0°, +- 15,30,45,60 around theta_init.
    const std::vector<double> deltas = {
        PI / 2.0,
        PI*5.0/12.0, -PI*5.0/12.0, //75
        PI / 3.0,  -PI / 3.0,   // 60°
        PI / 4.0,  -PI / 4.0,   // 45°
        PI / 6.0,  -PI / 6.0,   // 30°
        PI / 12.0, -PI / 12.0   // 15
    };

    double best_err = std::numeric_limits<double>::infinity();
    double best_theta = wrap_angle_pi(theta_init);

    for (double dtheta : deltas) {
        double theta = wrap_angle_pi(theta_init + dtheta);

        // Project points onto the line direction u = (cos(-theta), sin(-theta))
        // (equivalently: rotate axis by -theta).
        const double c = std::cos(-theta);
        const double s = std::sin(-theta);

        std::vector<double> t(points.size());
        for (int i = 0; i < (int)points.size(); ++i) {
            t[i] = points[i].x * c + points[i].y * s;
        }

        // Run your 1D solver (Jacobi + damping etc.) on this initialization
        double err = coefficient_descent_on_line(edges, adj_list, t);


        if (err < best_err) {
            best_err = err;
            best_theta = theta;
        }
    }

    std::cerr << "Best theta = " << best_theta << " theta star " << theta_init << ", best error = " << best_err << "\n";
    return best_err;
}


double optimized_projection_minErrDGP1_UB(
    const std::vector<Edge>& edges,
    const std::set<int>& vertex_ids
){
    //this function creates a UB for the MinerrDGP1 using random scatterings of the points onto the line and then performing the coefficient descent algorithm
    // Build adjacency list once
    auto adj_list = create_adjacency_list_from_edges(edges, edges.size());

    const int n = static_cast<int>(vertex_ids.size());
    if (n == 0) return 0.0;

    // ---- estimate a reasonable scale from the edges ----
    double avg_d = 0.0;
    for (const auto& e : edges) avg_d += e.weight;
    avg_d /= std::max(1, (int)edges.size());

    const int num_trials = 10;          // number of random restarts
    const double spread = 2.0 * avg_d;  // scale of random scattering
    const double noise = 0.1 * avg_d;   // small perturbation

    double best_error = std::numeric_limits<double>::infinity();

    // Convert vertex_ids to vector for indexing
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

            int u_idx = std::distance(vids.begin(),
                                      std::find(vids.begin(), vids.end(), u));

            for (const auto& nbr : adj_list[u]) {
                int v = nbr.neighbourId;
                if (visited[v]) continue;

                int v_idx = std::distance(vids.begin(),
                                          std::find(vids.begin(), vids.end(), v));

                // random sign
                double sign = (Random::get(0, 1) == 0) ? -1.0 : 1.0;
                t[v_idx] = t[u_idx] + sign * nbr.dist;

                // small noise to avoid symmetry lock-in
                t[v_idx] += Random::get<double>(-noise, noise);

                visited[v] = true;
                stack.push_back(v);
            }
        }

        // For disconnected vertices, scatter randomly
        for (int i = 0; i < n; ++i) {
            if (!visited[vids[i]]) {
                t[i] = Random::get<double>(-spread, spread);
            }
        }

        // ---- run the 1D solver ----
        double err = coefficient_descent_on_line(edges, adj_list, t);

        best_error = std::min(best_error, err);
    }

    return best_error;
}


// ------------------------------ all of the things relating to the relaxations come here ---------------------------


std::vector<Line> getLines(){
    //for now I will fix the lines we are projecting on but I definitely should make this more modularisable later, probably asking the user what he wants
    Line x_axis = Line(Point(0,0), 1, 0);
    Line y_axis = Line(Point(0,0), 0, 1);
    Line symmetry = Line(Point(0,0), 1, 1);

    return {x_axis, y_axis, symmetry};
}



double simple_projection_relax(const std::vector<Edge>& edges, const std::vector<Point>& points){
    std::vector<Line> lines = getLines();

    std::cerr << "Computing the relaxation on " << lines.size() << " lines \n";


    double tot_err = 0.0;
    for(auto& [i1, i2, d, i]: edges){
        double min_err = std::numeric_limits<double>::max();
        for(Line& line: lines){
            
            //have to decrement the vertex ids by one to map them correctly back to their indices in the point vector
            double proj_err = line.projection_error(points[i1-1], points[i2-1]);
            if(proj_err < min_err){
                min_err = proj_err; 
            }

        }
        tot_err += min_err;
    }
    return tot_err;
}


/*
I will give up on the geometric relaxation idea for now, I will come back to it later on
I have some cases where the rotational relaxation is lower than the total relaxation and both are higher than the upper bound (not good)
I still think there is an issue with the code as teh rotational relax should never be higher than the value given by the total relax
as we are rotating all the lines and all the points by the same angle and then adding back the line which we wanting to rotate all our points to
so normally we should be having the same setup at the total relaxation (just rotated) plus an additional line which is the original x axis line
*/
double rotation_projection_relax(const std::vector<Edge>& edges, std::vector<Point>& points){
    std::vector<Line> lines = getLines();

    double theta = optimal_rotation(edges, lines[0], points);
    std::cerr << "The optimal rotation is " << theta << "\n";

    //rotate all of the points using this value of theta
    double c = cos(theta);
    double s = sin(theta);
    for(auto& p: points){
        //use the 2D rotation matrix to do this
        double x_new = c*p.x - s*p.y;
        double y_new = s*p.x + c*p.y;
        p.x = x_new;
        p.y = y_new;
    }

    std::vector<Line> new_lines;
    //rotate all of the lines using this theta
    for(int i=0; i<static_cast<int>(lines.size()); ++i){
        double x_dir_new = c*lines[i].dir.x - s*lines[i].dir.y;
        double y_dir_new = s*lines[i].dir.x + c*lines[i].dir.y; 
        new_lines.push_back(Line(Point(0,0), x_dir_new,y_dir_new));
    }
    
    new_lines.push_back(Line(Point(0,0), 1, 0));

    std::cerr << "Computing the relaxation on " << lines.size() << " lines \n";

    double tot_err = 0.0;
    for(auto& [i1, i2, d, i]: edges){
        double min_err = std::numeric_limits<double>::max();
        for(Line& line: new_lines){
            
            //have to decrement the vertex ids by one to map them correctly back to their indices in the point vector
            double proj_err = line.projection_error(points[i1-1], points[i2-1]);
            if(proj_err < min_err){
                min_err = proj_err; 
            }

        }
        tot_err += min_err;
    }
    return tot_err;
}


//this function takes a cycle basis and the associated error (weight) of that cycle, the point is to obtain the best possible error
//by greedily selecting the cycle that are edge disjoint and have the greatest error
double greedy_packing_cycle_err(const std::vector<std::vector<std::vector<Edge>>>& cycle_basis,
    const std::vector<std::vector<double>>& cycle_errors){

    //first flatten the cycle into 1dimension
    std::vector<WeightedCycle> all_cycles;
    for(int i=0; i<static_cast<int>(cycle_basis.size()); ++i){
        for(int j=0; j<static_cast<int>(cycle_basis[i].size()); ++j){
            all_cycles.push_back({cycle_basis[i][j], cycle_errors[i][j]});
        }
    }

    //sort in decreasing order by weight (to have the cycles contributing to the error the most at the beginning)
    std::sort(all_cycles.begin(), all_cycles.end(),
            [](const WeightedCycle& a, const WeightedCycle& b) {
                return a.error > b.error;
            });

    std::set<std::pair<int,int>> used_edges;
    double total_lb = 0.0;

    for (const auto& cycle : all_cycles) {

        bool disjoint = true;

        for (const Edge& e : cycle.edges) {
            auto key = std::make_pair(std::min(e.u, e.v),
                                    std::max(e.u, e.v));
            if (used_edges.count(key)) {
                disjoint = false;
                break;
            }
        }

        if (disjoint) {
            total_lb += cycle.error;

            for (const Edge& e : cycle.edges) {
                auto key = std::make_pair(std::min(e.u, e.v),
                                        std::max(e.u, e.v));
                used_edges.insert(key);
            }
        }
    }

    return total_lb;
    
}
    