#include "mpc_tracker_cpp/vehicle_model.hpp"
#include <cmath>
#include <algorithm>

VehicleState integrate_kinematic(
    const VehicleState& s,
    const VehicleControl& u,
    const VehicleParams& p,
    double dt)
{
    VehicleControl uc = clamp_control(u, p);
    VehicleState next;
    next.x   = s.x   + s.v * std::cos(s.psi) * dt;
    next.y   = s.y   + s.v * std::sin(s.psi) * dt;
    next.psi = s.psi + s.v / p.wheelbase * std::tan(uc.delta) * dt;
    next.v   = std::clamp(s.v + uc.a * dt, p.min_speed, p.max_speed);
    return next;
}

VehicleControl clamp_control(const VehicleControl& u, const VehicleParams& p)
{
    VehicleControl c;
    c.a     = std::clamp(u.a,     p.max_decel, p.max_accel);
    c.delta = std::clamp(u.delta, -p.max_steer, p.max_steer);
    return c;
}
