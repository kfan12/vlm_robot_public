#pragma once
#include <Eigen/Core>

// Kinematic bicycle model state: [x, y, psi, v]
// Control: [a, delta]
struct VehicleState {
    double x   = 0.0;
    double y   = 0.0;
    double psi = 0.0;  // heading (yaw)
    double v   = 0.0;  // speed
};

struct VehicleControl {
    double a     = 0.0;  // acceleration
    double delta = 0.0;  // steering angle
};

struct VehicleParams {
    double wheelbase  = 0.32;
    double max_steer  = 0.45;
    double max_speed  = 1.2;
    double min_speed  = 0.0;
    double max_accel  = 1.2;
    double max_decel  = -1.8;
};

// Advance state by dt using Euler integration of kinematic bicycle model
VehicleState integrate_kinematic(
    const VehicleState& s,
    const VehicleControl& u,
    const VehicleParams& p,
    double dt);

// Clamp control to vehicle limits
VehicleControl clamp_control(const VehicleControl& u, const VehicleParams& p);
