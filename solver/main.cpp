#include "dgp_bnb.h"
#include <chrono>

int main(int argc, char* argv[]){
    if(argc != 3){
        std::cerr << "Usage " << argv[0] << "<dat_file.dat> <real/int weights (0 for real 1 for int)>";
        return 1;
    }

    if (std::stoi(argv[2]) != 0 && std::stoi(argv[2]) != 1) {
        std::cerr << "Usage: " << argv[0] 
                << " <dat_file.dat> <real/int weights (0 for real, 1 for int)>\n";
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    auto p_solver = solve_minerr_dgp1(argv[1], std::stoi(argv[2]));

    //p_solver = {minErr, bestEmbedding}
    
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t = t1-t0;
    std::cout << p_solver.first << " " << t.count();
}