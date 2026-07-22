// path_marker_node — mirrors a ROS nav_msgs/Path into the Gazebo Fortress
// 3D scene by drawing a TRIANGLE_STRIP ribbon through the Gazebo /marker service.
//
// The ROS path (/vlm_path_truth) is published in the `odom` frame; Gazebo markers are
// drawn in the world frame. Instead of a hardcoded spawn offset, the odom->world
// transform is computed live from the robot's ACTUAL pose: its Gazebo world pose
// (from /world/<world>/pose/info) vs its /odom pose. The static offset_* params
// are kept only as a fallback until both poses have been received.

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <ignition/transport/Node.hh>
#include <ignition/msgs/marker.pb.h>
#include <ignition/msgs/pose_v.pb.h>

#include <array>
#include <cmath>
#include <mutex>
#include <vector>

class PathMarkerNode : public rclcpp::Node
{
public:
    PathMarkerNode() : Node("path_marker_node")
    {
        // odom-frame origin expressed in the Gazebo world frame
        // (defaults match spawn_robot.launch.py).
        offset_x_ = this->declare_parameter("offset_x", -3.0);
        offset_y_ = this->declare_parameter("offset_y", 0.0);
        offset_z_ = this->declare_parameter("offset_z", 0.05); // lift above ground
        offset_yaw_ = this->declare_parameter("offset_yaw", 0.0);

        // Ribbon width in metres. Gazebo LINE_STRIP markers ignore width, so
        // the path is drawn as a flat TRIANGLE_STRIP ribbon of this width.
        line_width_ = this->declare_parameter("line_width", 0.050);
        marker_ns_ = this->declare_parameter("marker_ns", std::string("vlm_path_truth"));
        marker_id_ = this->declare_parameter("marker_id", 1);
        marker_service_ = this->declare_parameter("marker_service", std::string("/marker"));
        resend_period_s_ = this->declare_parameter("resend_period_s", 2.0);
        path_topic_ = this->declare_parameter("path_topic", std::string("/vlm_path_truth"));
        // Ribbon colour (RGB 0..1), default green.
        color_r_ = this->declare_parameter("color_r", 0.0);
        color_g_ = this->declare_parameter("color_g", 1.0);
        color_b_ = this->declare_parameter("color_b", 0.0);

        // Live odom->world transform from the robot's actual pose.
        auto_offset_ = this->declare_parameter("auto_offset", true);
        world_name_ = this->declare_parameter("world_name", std::string("cone_lane"));
        model_name_ = this->declare_parameter("model_name", std::string("robotcar"));
        // /vlm_path_truth is anchored in the ground-truth odom frame, so the transform
        // must use the SAME (ground-truth) odom pose, else the ribbon drifts.
        odom_topic_ = this->declare_parameter("odom_topic", std::string("/odom_truth"));
        // Lock the transform once at startup (false) or re-sample every path
        // callback (true). In simulation the world->odom offset is constant, so
        // continuous tracking only adds async-sampling jitter; leave it false.
        track_ = this->declare_parameter("track_continuously", false);
        smooth_alpha_ = this->declare_parameter("smooth_alpha", 0.1);
        pos_tol_ = this->declare_parameter("pos_tol_m", 0.02);

        last_send_ = this->now();

        sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic_, 10,
            [this](nav_msgs::msg::Path::SharedPtr msg)
            { on_path(msg); });

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10,
            [this](nav_msgs::msg::Odometry::SharedPtr msg)
            { on_odom(msg); });

        // Subscribe to all Gazebo entity poses to track the robot's world pose.
        if (auto_offset_)
        {
            const std::string pose_topic = "/world/" + world_name_ + "/pose/info";
            node_.Subscribe(pose_topic, &PathMarkerNode::on_gz_pose, this);
            RCLCPP_INFO(get_logger(), "auto_offset: tracking '%s' on %s",
                        model_name_.c_str(), pose_topic.c_str());
        }

        RCLCPP_INFO(get_logger(),
                    "path_marker_node ready: drawing /vlm_path_truth to Gazebo %s "
                    "(fallback offset x=%.2f y=%.2f z=%.2f yaw=%.2f)",
                    marker_service_.c_str(), offset_x_, offset_y_, offset_z_, offset_yaw_);
    }

private:
    void on_path(const nav_msgs::msg::Path::SharedPtr msg)
    {
        ignition::msgs::Marker marker;
        marker.set_ns(marker_ns_);
        marker.set_id(static_cast<uint64_t>(marker_id_));
        marker.set_action(ignition::msgs::Marker::ADD_MODIFY);
        // TRIANGLE_STRIP ribbon — gives a width-adjustable "thick line".
        // (LINE_STRIP ignores width and always renders as a 1px line.)
        marker.set_type(ignition::msgs::Marker::TRIANGLE_STRIP);
        marker.set_visibility(ignition::msgs::Marker::GUI);

        // Identity pose. The proto default orientation is (0,0,0,0) — an INVALID
        // quaternion that Gazebo applies to the whole marker, tilting the ribbon
        // out of the ground plane (looks like a shape floating with a line down
        // to the ground). Must set w=1 explicitly.
        auto *pose = marker.mutable_pose();
        pose->mutable_position()->set_x(0.0);
        pose->mutable_position()->set_y(0.0);
        pose->mutable_position()->set_z(0.0);
        pose->mutable_orientation()->set_x(0.0);
        pose->mutable_orientation()->set_y(0.0);
        pose->mutable_orientation()->set_z(0.0);
        pose->mutable_orientation()->set_w(1.0);

        // Line width (scale.x for LINE_STRIP).
        marker.mutable_scale()->set_x(1.0);
        marker.mutable_scale()->set_y(1.0);
        marker.mutable_scale()->set_z(1.0);

        // Ribbon colour (param-driven).
        const float r = static_cast<float>(color_r_);
        const float g = static_cast<float>(color_g_);
        const float b = static_cast<float>(color_b_);
        auto *mat = marker.mutable_material();
        set_color(mat->mutable_ambient(), r, g, b, 1.0f);
        set_color(mat->mutable_diffuse(), r, g, b, 1.0f);
        set_color(mat->mutable_emissive(), r, g, b, 1.0f);

        // Resolve the odom->world transform (tx, ty, yaw). When the robot's
        // actual world and odom poses are both known, derive it from them so the
        // ribbon always lines up with the real vehicle; otherwise fall back to
        // the static offset_* params.
        double tx, ty, yaw;
        {
            std::lock_guard<std::mutex> lk(pose_mtx_);
            if (auto_offset_ && have_world_ && have_odom_)
            {
                // Instantaneous odom->world transform from the actual poses:
                // world_robot = R(yaw)*odom_robot + t  =>  t = world - R*odom.
                const double iyaw = wyaw_ - oyaw_;
                const double c = std::cos(iyaw), s = std::sin(iyaw);
                const double itx = wx_ - (ox_ * c - oy_ * s);
                const double ity = wy_ - (ox_ * s + oy_ * c);

                if (!tf_inited_)
                {
                    tx_ = itx;
                    ty_ = ity;
                    tyaw_ = iyaw;
                    tf_inited_ = true;
                }
                else if (track_)
                {
                    // Exponential smoothing; lerp yaw on the shortest arc.
                    double dyaw = iyaw - tyaw_;
                    while (dyaw > M_PI)
                        dyaw -= 2 * M_PI;
                    while (dyaw < -M_PI)
                        dyaw += 2 * M_PI;
                    tx_ += smooth_alpha_ * (itx - tx_);
                    ty_ += smooth_alpha_ * (ity - ty_);
                    tyaw_ += smooth_alpha_ * dyaw;
                }
                tx = tx_;
                ty = ty_;
                yaw = tyaw_;
            }
            else
            {
                tx = offset_x_;
                ty = offset_y_;
                yaw = offset_yaw_;
            }
        }
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);

        // First transform the path into world-frame centerline points.
        std::vector<std::array<double, 3>> centers;
        centers.reserve(msg->poses.size());
        for (const auto &ps : msg->poses)
        {
            const double px = ps.pose.position.x;
            const double py = ps.pose.position.y;
            const double pz = ps.pose.position.z;
            centers.push_back({
                tx + px * cy - py * sy,
                ty + px * sy + py * cy,
                offset_z_ + pz,
            });
        }

        if (centers.size() < 2)
            return; // nothing meaningful to draw yet

        // The path is usually static (e.g. the locked circle), but it arrives at
        // 5 Hz. Re-sending the marker every time forces Gazebo to rebuild the
        // whole ribbon mesh each frame, which stalls the render thread and makes
        // the moving robot visual flicker back to a stale pose. So only send when
        // the path actually changed, with a slow heartbeat resend so a GUI opened
        // later still receives it.
        const auto now = this->now();
        const bool stale = (now - last_send_).seconds() > resend_period_s_;
        if (!centers_changed(centers, last_centers_, pos_tol_) && !stale)
            return;
        last_centers_ = centers;
        last_send_ = now;

        // Build a flat ribbon: at each centerline point emit a left and right
        // edge vertex offset by half-width along the local normal. Consecutive
        // L/R vertices form the TRIANGLE_STRIP.
        const double half = 0.5 * line_width_;
        const size_t n = centers.size();
        for (size_t i = 0; i < n; ++i)
        {
            // Tangent from neighbouring points (forward/backward at the ends).
            const auto &prev = centers[i == 0 ? 0 : i - 1];
            const auto &next = centers[i == n - 1 ? n - 1 : i + 1];
            double tx = next[0] - prev[0];
            double ty = next[1] - prev[1];
            const double len = std::hypot(tx, ty);
            if (len > 1e-9)
            {
                tx /= len;
                ty /= len;
            }
            // Left-hand normal (-ty, tx).
            const double nx = -ty * half;
            const double ny = tx * half;

            auto *left = marker.add_point();
            left->set_x(centers[i][0] + nx);
            left->set_y(centers[i][1] + ny);
            left->set_z(centers[i][2]);

            auto *right = marker.add_point();
            right->set_x(centers[i][0] - nx);
            right->set_y(centers[i][1] - ny);
            right->set_z(centers[i][2]);
        }

        // One-way request to the Gazebo marker service.
        node_.Request(marker_service_, marker);
    }

    // True if any point moved more than tol, or the point count changed.
    static bool centers_changed(const std::vector<std::array<double, 3>> &a,
                                const std::vector<std::array<double, 3>> &b,
                                double tol)
    {
        if (a.size() != b.size())
            return true;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]) > tol ||
                std::abs(a[i][2] - b[i][2]) > tol)
                return true;
        return false;
    }

    static void set_color(ignition::msgs::Color *c, float r, float g, float b, float a)
    {
        c->set_r(r);
        c->set_g(g);
        c->set_b(b);
        c->set_a(a);
    }

    static double quat_yaw(double x, double y, double z, double w)
    {
        return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    }

    // Robot pose in the odom frame (ROS /odom).
    void on_odom(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        const auto &p = msg->pose.pose.position;
        const auto &q = msg->pose.pose.orientation;
        std::lock_guard<std::mutex> lk(pose_mtx_);
        ox_ = p.x;
        oy_ = p.y;
        oyaw_ = quat_yaw(q.x, q.y, q.z, q.w);
        have_odom_ = true;
    }

    // Robot pose in the world frame (Gazebo pose/info — runs on an ign thread).
    void on_gz_pose(const ignition::msgs::Pose_V &msg)
    {
        for (const auto &p : msg.pose())
        {
            if (p.name() != model_name_)
                continue;
            const auto &q = p.orientation();
            std::lock_guard<std::mutex> lk(pose_mtx_);
            wx_ = p.position().x();
            wy_ = p.position().y();
            wyaw_ = quat_yaw(q.x(), q.y(), q.z(), q.w());
            have_world_ = true;
            return;
        }
    }

    double offset_x_ = -3.0;
    double offset_y_ = 0.0;
    double offset_z_ = 0.05;
    double offset_yaw_ = 0.0;
    double line_width_ = 0.03;
    int marker_id_ = 1;
    std::string marker_ns_ = "vlm_path_truth";
    std::string marker_service_ = "/marker";
    double resend_period_s_ = 2.0;
    std::string path_topic_ = "/vlm_path_truth";
    double color_r_ = 0.0, color_g_ = 1.0, color_b_ = 0.0;

    // Live odom->world transform inputs.
    bool auto_offset_ = true;
    std::string world_name_ = "cone_lane";
    std::string model_name_ = "robotcar";
    std::string odom_topic_ = "/odom_truth";
    std::mutex pose_mtx_;
    double wx_ = 0.0, wy_ = 0.0, wyaw_ = 0.0;
    bool have_world_ = false;
    double ox_ = 0.0, oy_ = 0.0, oyaw_ = 0.0;
    bool have_odom_ = false;
    bool track_ = true;
    double smooth_alpha_ = 0.1;
    double pos_tol_ = 0.02;
    bool tf_inited_ = false;
    double tx_ = 0.0, ty_ = 0.0, tyaw_ = 0.0; // smoothed odom->world transform

    std::vector<std::array<double, 3>> last_centers_;
    rclcpp::Time last_send_;

    ignition::transport::Node node_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathMarkerNode>());
    rclcpp::shutdown();
    return 0;
}
