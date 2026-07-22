// path_relay_node — relocate one OR MORE paths from a source odom frame into a
// destination odom frame, so the controller can localize on one frame while the
// path is drawn/visualized against another.
//
// Subscribes:
//   <each path_in_topics[i]>  (nav_msgs/Path)      — path in the source odom frame
//   <odom_src_topic>          (nav_msgs/Odometry)  — robot pose in the source frame
//   <odom_dest_topic>         (nav_msgs/Odometry)  — robot pose in the dest frame
// Publishes:
//   <each path_out_topics[i]> (nav_msgs/Path)      — same path expressed in the dest frame
//
// The two odom frames share the spawn origin but diverge by the accumulated
// drift. The relocation is the rigid transform T = destPose o srcPose^-1,
// sampled (locked) at the instant each path arrives. All path pairs share the
// single src/dest odom pair; each keeps its OWN publisher and last-relayed path
// so the periodic re-publish covers every path, not just the most recent.

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

class PathRelayNode : public rclcpp::Node
{
public:
    PathRelayNode() : Node("path_relay_node")
    {
        auto in_topics = this->declare_parameter(
            "path_in_topics", std::vector<std::string>{"/vlm_path_truth"});
        auto out_topics = this->declare_parameter(
            "path_out_topics", std::vector<std::string>{"/vlm_path_odom"});
        odom_src_topic_ = this->declare_parameter("odom_src_topic", std::string("/odom_truth"));
        odom_dest_topic_ = this->declare_parameter("odom_dest_topic", std::string("/odom_ekf"));
        out_frame_ = this->declare_parameter("out_frame_id", std::string("odom"));

        if (in_topics.size() != out_topics.size())
        {
            throw std::runtime_error(
                "path_in_topics and path_out_topics must have the same length");
        }

        odom_src_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_src_topic_, 10,
            [this](nav_msgs::msg::Odometry::SharedPtr msg)
            { odom_src_ = msg; });
        odom_dest_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_dest_topic_, 10,
            [this](nav_msgs::msg::Odometry::SharedPtr msg)
            { odom_dest_ = msg; });

        // One subscription + publisher + last-relayed buffer per path pair.
        for (size_t i = 0; i < in_topics.size(); ++i)
        {
            auto pair = std::make_shared<Pair>();
            pair->in = in_topics[i];
            pair->out = out_topics[i];
            pair->pub = this->create_publisher<nav_msgs::msg::Path>(pair->out, 10);
            Pair *raw = pair.get(); // capture raw pointer for lambda to avoid shared_ptr cycle
            pair->sub = this->create_subscription<nav_msgs::msg::Path>(
                pair->in, 10,
                [this, raw](nav_msgs::msg::Path::SharedPtr msg)
                { on_path(raw, msg); });
            pairs_.push_back(pair);
        }

        republish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200),
            [this]()
            { republish(); });
    }

private:
    struct Pair
    {
        std::string in, out;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub;
        rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub;
        nav_msgs::msg::Path::SharedPtr last_out;
    };

    // Return the yaw angle (radians) of a quaternion.
    static double quat_yaw(const geometry_msgs::msg::Quaternion &q)
    {
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }

    void on_path(Pair *pair, const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (!odom_src_ || !odom_dest_)
        {
            RCLCPP_WARN(get_logger(),
                        "Got %s but missing %s/%s — not relaying yet.",
                        pair->in.c_str(), odom_src_topic_.c_str(), odom_dest_topic_.c_str());
            return;
        }

        // Lock the relocation transform at path-receive time.
        const double gx = odom_src_->pose.pose.position.x;
        const double gy = odom_src_->pose.pose.position.y;
        const double gth = quat_yaw(odom_src_->pose.pose.orientation);
        const double ox = odom_dest_->pose.pose.position.x;
        const double oy = odom_dest_->pose.pose.position.y;
        const double oth = quat_yaw(odom_dest_->pose.pose.orientation);

        const double dth = oth - gth;
        const double c = std::cos(dth), s = std::sin(dth);

        nav_msgs::msg::Path out;
        out.header.stamp = this->get_clock()->now();
        out.header.frame_id = out_frame_;
        out.poses.reserve(msg->poses.size());

        for (const auto &ps : msg->poses) // each pose in the source path
        {
            const double rx = ps.pose.position.x - gx;
            const double ry = ps.pose.position.y - gy;

            geometry_msgs::msg::PoseStamped p;
            p.header = out.header;
            p.pose.position.x = ox + (c * rx - s * ry);
            p.pose.position.y = oy + (s * rx + c * ry);
            p.pose.position.z = ps.pose.position.z;

            // Rotate the pose heading by the same delta.
            const double yaw = quat_yaw(ps.pose.orientation) + dth;
            // Convert back to quaternion.
            p.pose.orientation.x = 0.0;
            p.pose.orientation.y = 0.0;
            p.pose.orientation.z = std::sin(yaw / 2.0);
            p.pose.orientation.w = std::cos(yaw / 2.0);
            out.poses.push_back(p);
        }

        pair->last_out = std::make_shared<nav_msgs::msg::Path>(out);
        pair->pub->publish(out);
        // RCLCPP_INFO(get_logger(),
        //             "Relayed %zu poses %s -> %s (drift dx=%.3f dy=%.3f dyaw=%.3f).",
        //             out.poses.size(), pair->in.c_str(), pair->out.c_str(),
        //             ox - gx, oy - gy, dth);
    }

    // Periodically re-publish the last-relayed path for each pair, so the visualization
    // doesn't disappear if the source path stops publishing.
    void republish()
    {
        for (auto &pair : pairs_)
        {
            if (pair->last_out)
            {
                pair->last_out->header.stamp = this->get_clock()->now();
                pair->pub->publish(*pair->last_out);
            }
        }
    }

    std::string odom_src_topic_, odom_dest_topic_, out_frame_;
    nav_msgs::msg::Odometry::SharedPtr odom_src_, odom_dest_;
    std::vector<std::shared_ptr<Pair>> pairs_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_src_sub_, odom_dest_sub_;
    rclcpp::TimerBase::SharedPtr republish_timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathRelayNode>());
    rclcpp::shutdown();
    return 0;
}
