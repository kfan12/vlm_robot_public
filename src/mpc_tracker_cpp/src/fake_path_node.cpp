#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>

class FakePathNode : public rclcpp::Node
{
public:
    FakePathNode() : Node("fake_path_node")
    {
        radius_ = this->declare_parameter("radius_m", 2.0);
        n_pts_ = this->declare_parameter("n_points", 80);

        pub_ = this->create_publisher<nav_msgs::msg::Path>("/vlm_path_truth", 10);
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            [this](nav_msgs::msg::Odometry::SharedPtr msg)
            { on_odom(msg); });
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200),
            std::bind(&FakePathNode::publish_path, this));
    }

private:
    void on_odom(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        odom_ = msg;
        if (!circle_locked_)
            lock_circle();
    }

    // Lock circle center on first odom — center is radius_ to the left of robot
    void lock_circle()
    {
        double ox = odom_->pose.pose.position.x;
        double oy = odom_->pose.pose.position.y;
        double oyaw = tf2::getYaw(odom_->pose.pose.orientation);

        cx_ = ox - radius_ * std::sin(oyaw);
        cy_ = oy + radius_ * std::cos(oyaw);
        // Angle from center to robot (robot starts on the circle)
        start_angle_ = std::atan2(oy - cy_, ox - cx_);
        circle_locked_ = true;
        RCLCPP_INFO(get_logger(), "Circle locked: center=(%.2f, %.2f) radius=%.2f",
                    cx_, cy_, radius_);
    }

    void publish_path()
    {
        if (!circle_locked_)
            return;

        nav_msgs::msg::Path path;
        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = "odom";

        for (int i = 0; i <= n_pts_; ++i)
        {
            double angle = start_angle_ + 2.0 * M_PI * i / n_pts_;

            geometry_msgs::msg::PoseStamped ps;
            ps.header = path.header;
            ps.pose.position.x = cx_ + radius_ * std::cos(angle);
            ps.pose.position.y = cy_ + radius_ * std::sin(angle);
            ps.pose.position.z = 0.0;

            // Tangent direction (counterclockwise)
            tf2::Quaternion q;
            q.setRPY(0, 0, angle + M_PI / 2.0);
            ps.pose.orientation = tf2::toMsg(q);
            path.poses.push_back(ps);
        }
        pub_->publish(path);
    }

    double radius_ = 2.0;
    int n_pts_ = 80;
    double cx_ = 0.0;
    double cy_ = 0.0;
    double start_angle_ = 0.0;
    bool circle_locked_ = false;

    nav_msgs::msg::Odometry::SharedPtr odom_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FakePathNode>());
    rclcpp::shutdown();
    return 0;
}
