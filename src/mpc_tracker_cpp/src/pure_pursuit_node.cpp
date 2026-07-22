#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

class PurePursuitNode : public rclcpp::Node
{
public:
    PurePursuitNode() : Node("pure_pursuit_node")
    {
        wheelbase_ = this->declare_parameter("wheelbase_m", 0.32);
        lookahead_ = this->declare_parameter("lookahead_m", 0.8);
        target_speed_ = this->declare_parameter("target_speed_mps", 0.4);
        max_steer_ = this->declare_parameter("max_steer_rad", 0.45);
        // Stop at the path end: brake within slowdown_radius, halt within goal_tol.
        goal_tol_ = this->declare_parameter("goal_tolerance_m", 0.25);
        slowdown_radius_ = this->declare_parameter("slowdown_radius_m", 1.0);
        // Track the path relocated into the wheel-odom frame (path_relay_node).
        // Use "/vlm_path_truth" directly for the old single-frame flow.

        // Path + localization source. Default control flow uses the EKF estimate
        // /odom_ekf and the path relocated into that frame (/vlm_path_odom).
        // Path and odom MUST be in the same frame:
        //   /vlm_path_odom + /odom_ekf   (EKF control, default)
        //   /vlm_path_odom + /odom       (raw wheel-odom control)
        //   /vlm_path_truth      + /odom_truth (ground-truth control)
        std::string path_topic = this->declare_parameter(
            "path_topic", std::string("/vlm_path_odom"));
        std::string odom_topic = this->declare_parameter(
            "odom_topic", std::string("/odom_ekf"));

        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic, 10,
            [this](nav_msgs::msg::Path::SharedPtr msg)
            {
                path_ = msg;
                arrived_ = false; // new path: drive again until its end
            });

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, 10,
            [this](nav_msgs::msg::Odometry::SharedPtr msg)
            { odom_ = msg; });

        RCLCPP_INFO(get_logger(), "pure_pursuit: path=%s odom=%s",
                    path_topic.c_str(), odom_topic.c_str());

        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&PurePursuitNode::control_tick, this));
    }

private:
    void control_tick()
    {
        if (!path_ || !odom_)
            return;
        if (path_->poses.empty())
        {
            stop();
            return;
        }

        // Stuck detection: if commanding motion but actual speed is near zero
        double actual_speed = std::hypot(odom_->twist.twist.linear.x,
                                         odom_->twist.twist.linear.y);
        if (commanding_ && actual_speed < 0.02)
        {
            stuck_ticks_++;
        }
        else
        {
            stuck_ticks_ = 0;
        }
        if (stuck_ticks_ > 20)
        { // ~1s at 20Hz
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Robot stuck — stopping");
            stop();
            return;
        }

        double rx = odom_->pose.pose.position.x;
        double ry = odom_->pose.pose.position.y;
        double rh = tf2::getYaw(odom_->pose.pose.orientation);

        // Stop at the end of the curve: brake near the final path point.
        const auto &last = path_->poses.back().pose.position;
        double dist_to_goal = std::hypot(last.x - rx, last.y - ry);
        if (arrived_ || dist_to_goal <= goal_tol_)
        {
            if (!arrived_)
                RCLCPP_INFO(get_logger(), "Reached end of path — stopping.");
            arrived_ = true;
            stop();
            return;
        }

        // Find the closest point on the path, then look ahead FORWARD from it.
        // (Searching from the path start would grab points behind the robot once
        // it has driven past them, making it wander instead of follow.)
        size_t closest = 0;
        double best = std::numeric_limits<double>::max();
        for (size_t i = 0; i < path_->poses.size(); ++i)
        {
            const auto &p = path_->poses[i].pose.position;
            double d = std::hypot(p.x - rx, p.y - ry);
            if (d < best)
            {
                best = d;
                closest = i;
            }
        }

        geometry_msgs::msg::PoseStamped goal = path_->poses.back();
        for (size_t i = closest; i < path_->poses.size(); ++i)
        {
            const auto &p = path_->poses[i].pose.position;
            if (std::hypot(p.x - rx, p.y - ry) >= lookahead_)
            {
                goal = path_->poses[i];
                break;
            }
        }

        // Compute pure pursuit steering
        double gx = goal.pose.position.x - rx;
        double gy = goal.pose.position.y - ry;
        double angle_to_goal = std::atan2(gy, gx) - rh;
        // Normalize to [-pi, pi]
        while (angle_to_goal > M_PI)
            angle_to_goal -= 2 * M_PI;
        while (angle_to_goal < -M_PI)
            angle_to_goal += 2 * M_PI;

        double dist = std::hypot(gx, gy);
        double curvature = 2.0 * std::sin(angle_to_goal) / (dist + 1e-9);
        double steer = std::atan(curvature * wheelbase_);
        steer = std::clamp(steer, -max_steer_, max_steer_);

        // Bang-bang: alternate full speed / near-stop every second.
        double now_sec = this->now().seconds();
        double speed = std::fmod(now_sec, 2.0) < 1.0 ? target_speed_ : 0.1;
        if (dist_to_goal < slowdown_radius_)
            speed = target_speed_ * std::clamp(dist_to_goal / slowdown_radius_,
                                               0.25, 1.0);

        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = speed;
        cmd.angular.z = steer * speed / wheelbase_;
        cmd_pub_->publish(cmd);
        commanding_ = true;
    }

    void stop()
    {
        geometry_msgs::msg::Twist cmd;
        cmd_pub_->publish(cmd);
        commanding_ = false;
        stuck_ticks_ = 0;
    }

    double wheelbase_, lookahead_, target_speed_, max_steer_;
    double goal_tol_ = 0.25, slowdown_radius_ = 1.0;
    bool commanding_ = false;
    bool arrived_ = false;
    int stuck_ticks_ = 0;
    nav_msgs::msg::Path::SharedPtr path_;
    nav_msgs::msg::Odometry::SharedPtr odom_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuitNode>());
    rclcpp::shutdown();
    return 0;
}
