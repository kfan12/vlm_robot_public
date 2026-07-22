// ekf_sf_node — a hand-written 5-state Extended Kalman Filter that fuses the
// wheel-encoder forward speed (/odom twist.linear.x) with the IMU yaw +
// yaw-rate (/imu) into a fused odometry on /odom_sf.
//
// State:  x = [px, py, theta, v, omega]
// Predict: CTRV-style constant-velocity / constant-turn-rate kinematics.
// Update:  wheel encoder -> v ;  IMU -> [theta, omega].
//
// TF is OFF by default: the stock robot_localization node (ekf.yaml) already
// owns odom->base_link. Enable publish_tf only if running this node standalone.

#include <cmath>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include <Eigen/Dense>

namespace
{
  // Wrap an angle to (-pi, pi].
  double wrap_angle(double a)
  {
    while (a > M_PI)
      a -= 2.0 * M_PI;
    while (a < -M_PI)
      a += 2.0 * M_PI;
    return a;
  }
} // namespace

class EkfSfNode : public rclcpp::Node
{
public:
  EkfSfNode()
      : rclcpp::Node("ekf_sf_node")
  {
    // ---- Parameters -------------------------------------------------------
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_link_frame", "base_link");
    publish_tf_ = declare_parameter<bool>("publish_tf", false);

    const std::string wheel_topic = declare_parameter<std::string>("wheel_topic", "/odom");
    const std::string imu_topic = declare_parameter<std::string>("imu_topic", "/imu");
    const std::string out_topic = declare_parameter<std::string>("output_topic", "/odom_sf");

    // Process noise: how much we let v and omega drift between predictions.
    q_accel_ = declare_parameter<double>("q_accel", 1.0);         // (m/s^2)^2 spectral
    q_yaw_accel_ = declare_parameter<double>("q_yaw_accel", 1.0); // (rad/s^2)^2 spectral
    q_pos_ = declare_parameter<double>("q_pos", 1e-4);            // small jitter on x,y
    q_theta_ = declare_parameter<double>("q_theta", 1e-4);        // small jitter on theta

    // Measurement noise (variances).
    r_wheel_v_ = declare_parameter<double>("r_wheel_v", 0.04);           // (m/s)^2
    r_imu_yaw_ = declare_parameter<double>("r_imu_yaw", 0.01);           // (rad)^2
    r_imu_yawrate_ = declare_parameter<double>("r_imu_yawrate", 0.0001); // (rad/s)^2

    // ---- Filter init ------------------------------------------------------
    x_.setZero();
    P_.setIdentity();
    P_ *= 1.0; // adjust P_

    // ---- ROS I/O ----------------------------------------------------------
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(out_topic, 10);

    wheel_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        wheel_topic, rclcpp::SensorDataQoS(),
        std::bind(&EkfSfNode::wheelCallback, this, std::placeholders::_1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, rclcpp::SensorDataQoS(),
        std::bind(&EkfSfNode::imuCallback, this, std::placeholders::_1));

    if (publish_tf_)
    {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    RCLCPP_INFO(
        get_logger(),
        "ekf_sf_node up: wheel='%s' imu='%s' -> '%s' (publish_tf=%s)",
        wheel_topic.c_str(), imu_topic.c_str(), out_topic.c_str(),
        publish_tf_ ? "true" : "false");
  }

private:
  using Vec5 = Eigen::Matrix<double, 5, 1>;
  using Mat5 = Eigen::Matrix<double, 5, 5>;

  // -- Indices into the state vector --
  enum
  {
    PX = 0,
    PY = 1,
    THETA = 2,
    V = 3,
    OMEGA = 4
  };

  // EKF prediction up to time `stamp` (CTRV kinematics).
  void predictTo(const rclcpp::Time &stamp)
  {
    if (!initialized_)
    {
      last_predict_time_ = stamp;
      initialized_ = true;
      return;
    }

    double dt = (stamp - last_predict_time_).seconds();
    if (dt <= 0.0)
    {
      // Out-of-order / duplicate stamp: skip propagation, keep last time.
      return;
    }
    // Guard against large gaps (e.g. paused sim) that would blow up the filter.
    if (dt > 1.0)
    {
      dt = 1.0;
    }
    last_predict_time_ = stamp;

    const double th = x_(THETA);
    const double v = x_(V);
    const double w = x_(OMEGA);
    const double c = std::cos(th);
    const double s = std::sin(th);

    // Nonlinear state propagation.
    x_(PX) += v * c * dt;
    x_(PY) += v * s * dt;
    x_(THETA) = wrap_angle(th + w * dt); // step1, update estimation:  x_t = f(x_t-1,u_t), u_t = (v_t, w_t)
    // v, omega assumed constant over dt.

    // Jacobian F = d f / d x.
    Mat5 F = Mat5::Identity();
    F(PX, THETA) = -v * s * dt;
    F(PX, V) = c * dt;
    F(PY, THETA) = v * c * dt;
    F(PY, V) = s * dt;
    F(THETA, OMEGA) = dt;

    // Process noise Q. v and omega gain variance from unmodeled accel.
    Mat5 Q = Mat5::Zero();
    Q(PX, PX) = q_pos_ * dt;
    Q(PY, PY) = q_pos_ * dt;
    Q(THETA, THETA) = q_theta_ * dt;
    Q(V, V) = q_accel_ * dt;
    Q(OMEGA, OMEGA) = q_yaw_accel_ * dt;

    P_ = F * P_ * F.transpose() + Q; // step2, update uncertaitny matrix
  }

  // Wheel encoder: measures forward speed v.
  void wheelCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const rclcpp::Time stamp(msg->header.stamp);
    predictTo(stamp);

    const double z = msg->twist.twist.linear.x; // 1*1 matrix => double

    Eigen::Matrix<double, 1, 5> H = Eigen::Matrix<double, 1, 5>::Zero();
    H(0, V) = 1.0;

    const double y = z - x_(V); // innovation
    const double S = (H * P_ * H.transpose())(0, 0) + r_wheel_v_;
    const Vec5 K = (P_ * H.transpose()) / S; // 5x1 gain   step3, Kalman gain

    x_ += K * y; // step4, correct state estimation
    x_(THETA) = wrap_angle(x_(THETA));
    P_ = (Mat5::Identity() - K * H) * P_; // step5, correct uncertainty matrix

    publish(stamp);
  }

  // IMU: measures absolute yaw (theta) and yaw-rate (omega).
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const rclcpp::Time stamp(msg->header.stamp);
    predictTo(stamp);

    const double yaw = tf2::getYaw(msg->orientation);
    const double wz = msg->angular_velocity.z;

    Eigen::Matrix<double, 2, 5> H = Eigen::Matrix<double, 2, 5>::Zero();
    H(0, THETA) = 1.0;
    H(1, OMEGA) = 1.0;

    Eigen::Vector2d z(yaw, wz);
    Eigen::Vector2d h(x_(THETA), x_(OMEGA));
    Eigen::Vector2d y = z - h;
    y(0) = wrap_angle(y(0)); // wrap yaw innovation

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = r_imu_yaw_;
    R(1, 1) = r_imu_yawrate_;

    Eigen::Matrix2d S = H * P_ * H.transpose() + R;
    Eigen::Matrix<double, 5, 2> K = P_ * H.transpose() * S.inverse(); // 5x2 gain, step3, Kalman gain

    x_ += K * y; // step4, correct state estimation
    x_(THETA) = wrap_angle(x_(THETA));
    P_ = (Mat5::Identity() - K * H) * P_; // step5, correct uncertainty matrix

    publish(stamp);
  }

  void publish(const rclcpp::Time &stamp)
  {
    nav_msgs::msg::Odometry out;
    out.header.stamp = stamp;
    out.header.frame_id = odom_frame_;
    out.child_frame_id = base_frame_;

    out.pose.pose.position.x = x_(PX);
    out.pose.pose.position.y = x_(PY);
    out.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, x_(THETA));
    out.pose.pose.orientation = tf2::toMsg(q);

    out.twist.twist.linear.x = x_(V);
    out.twist.twist.angular.z = x_(OMEGA);

    // Map the 5-state covariance into the 6-DOF pose/twist covariance blocks.
    // Pose order: x, y, z, roll, pitch, yaw.
    out.pose.covariance[0] = P_(PX, PX);        // x,x
    out.pose.covariance[1] = P_(PX, PY);        // x,y
    out.pose.covariance[6] = P_(PY, PX);        // y,x
    out.pose.covariance[7] = P_(PY, PY);        // y,y
    out.pose.covariance[35] = P_(THETA, THETA); // yaw,yaw
    // Twist order: vx, vy, vz, vroll, vpitch, vyaw.
    out.twist.covariance[0] = P_(V, V);          // vx,vx
    out.twist.covariance[35] = P_(OMEGA, OMEGA); // vyaw,vyaw

    odom_pub_->publish(out);

    if (publish_tf_ && tf_broadcaster_)
    {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.header.frame_id = odom_frame_;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = x_(PX);
      tf.transform.translation.y = x_(PY);
      tf.transform.translation.z = 0.0;
      tf.transform.rotation = out.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  // -- State --
  Vec5 x_;
  Mat5 P_;
  bool initialized_{false};
  rclcpp::Time last_predict_time_;

  // -- Params --
  std::string odom_frame_, base_frame_;
  bool publish_tf_{false};
  double q_accel_, q_yaw_accel_, q_pos_, q_theta_;
  double r_wheel_v_, r_imu_yaw_, r_imu_yawrate_;

  // -- ROS --
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EkfSfNode>());
  rclcpp::shutdown();
  return 0;
}
