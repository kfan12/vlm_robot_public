#pragma once
#include <vector>
#include <nav_msgs/msg/path.hpp>
#include "mpc_tracker_cpp/vehicle_model.hpp"

struct PathPoint {
    double x, y, psi, s;  // position, heading, arc-length
};

// Convert nav_msgs/Path to PathPoint vector
std::vector<PathPoint> path_msg_to_points(const nav_msgs::msg::Path& msg);

// Find the index of the closest path point to the current state
size_t find_closest_point(
    const std::vector<PathPoint>& path,
    const VehicleState& state,
    double search_radius = 2.0);

// Resample path uniformly by arc-length, starting from start_idx
std::vector<PathPoint> resample_path(
    const std::vector<PathPoint>& path,
    size_t start_idx,
    int N,       // number of horizon steps
    double ds);  // arc-length step
