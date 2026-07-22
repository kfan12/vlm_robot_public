#include "mpc_tracker_cpp/vehicle_model.hpp"
#include "mpc_tracker_cpp/path_utils.hpp"
#include "mpc_tracker_cpp/mpc_solver.hpp"
#include <iostream>
int main()
{
    MpcParams mp;
    VehicleParams vp;
    MpcSolver solver(vp, mp);
    VehicleState s;
    s.x = 0;
    s.y = 0;
    s.psi = 0;
    s.v = 0.3;
    std::vector<PathPoint> ref;
    for (int i = 0; i < 15; ++i)
    {
        PathPoint p;
        p.x = i * 0.05;
        p.y = 0;
        p.psi = 0;
        p.s = i * 0.05;
        ref.push_back(p);
    }
    auto result = solver.Solve(s, ref, 0.0, 0.0);
    std::cout << "Success: " << result.success << "\n";
    std::cout << "Steer: " << result.steer << "\n";
    std::cout << "Accel: " << result.accel << "\n";
    return 0;
}