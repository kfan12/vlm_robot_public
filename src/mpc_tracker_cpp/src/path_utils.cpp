#include "mpc_tracker_cpp/path_utils.hpp"
#include <cmath>
#include <limits>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

std::vector<PathPoint> path_msg_to_points(const nav_msgs::msg::Path &msg)
{
    std::vector<PathPoint> pts;
    double s = 0.0;
    for (size_t i = 0; i < msg.poses.size(); ++i)
    {
        const auto &pose = msg.poses[i].pose;
        PathPoint p;
        p.x = pose.position.x;                 // position x
        p.y = pose.position.y;                 // position y
        p.psi = tf2::getYaw(pose.orientation); // orientation yaw
        if (i > 0)
        {
            double dx = p.x - pts.back().x;
            double dy = p.y - pts.back().y;
            s += std::hypot(dx, dy);
        }
        p.s = s; // cumulative path length
        pts.push_back(p);
    }
    return pts;
}

size_t find_closest_point(
    const std::vector<PathPoint> &path,
    const VehicleState &state,
    double search_radius)
{
    size_t best = 0;
    double best_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < path.size(); ++i)
    {
        double dx = path[i].x - state.x;
        double dy = path[i].y - state.y;
        double d = std::hypot(dx, dy);
        if (d < best_dist && d < search_radius)
        {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

// resample the path from the closest pose idx
std::vector<PathPoint> resample_path(
    const std::vector<PathPoint> &path,
    size_t start_idx,
    int N,
    double ds)
{
    std::vector<PathPoint> ref;
    if (path.empty() || start_idx >= path.size())
        return ref;

    double s_start = path[start_idx].s;
    size_t j = start_idx;
    for (int i = 0; i < N; ++i)
    {
        double s_target = s_start + i * ds;
        while (j + 1 < path.size() && path[j + 1].s < s_target)
            ++j;
        if (j + 1 >= path.size())
        {
            ref.push_back(path.back());
        }
        else
        {
            double t = (s_target - path[j].s) / (path[j + 1].s - path[j].s + 1e-9);
            PathPoint interp;
            interp.x = path[j].x + t * (path[j + 1].x - path[j].x);
            interp.y = path[j].y + t * (path[j + 1].y - path[j].y);
            interp.psi = path[j].psi + t * (path[j + 1].psi - path[j].psi);
            interp.s = s_target;
            ref.push_back(interp);
        }
    }
    return ref;
}
