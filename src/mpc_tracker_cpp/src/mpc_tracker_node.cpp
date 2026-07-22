#include <rclcpp/rclcpp.hpp>
#include "mpc_tracker_cpp/mpc_solver.hpp"
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

class MpcTrackerNode : public rclcpp::Node
{
public:
    MpcTrackerNode() : Node("mpc_tracker")
    {
        // Parameters
        rate_hz_ = declare_parameter("control_rate_hz", 30.0);
        int N = declare_parameter("horizon_steps", 15);
        double dt = declare_parameter("dt", 0.1);
        double wb = declare_parameter("wheelbase_m", 0.32);
        double t_spd = declare_parameter("target_speed_mps", 0.6);
        double max_spd = declare_parameter("max_speed_mps", 1.2);
        double max_acc = declare_parameter("max_accel_mps2", 1.2);
        double max_dec = declare_parameter("max_decel_mps2", -1.8);
        double max_str = declare_parameter("max_steer_rad", 0.45);
        double w_lat = declare_parameter("weight_lateral_error", 8.0);
        double w_hd = declare_parameter("weight_heading_error", 3.0);
        double w_spd = declare_parameter("weight_speed_error", 1.0);
        double w_acc = declare_parameter("weight_accel", 0.2);
        double w_str = declare_parameter("weight_steer", 0.5);
        double w_acc_rate = declare_parameter("weight_accel_rate", 0.4);
        double w_str_rate = declare_parameter("weight_steer_rate", 2.0);
        double look_m = declare_parameter("heading_lookahead_m", 0.6);
        // Maneuver-dependent heading lookahead: the chord to the lookahead point
        // cuts inside curves by ~L^2/2R, so turns want it SHORT while straights
        // want it LONG (noise averaging). The planner publishes its maneuver FSM
        // state on maneuver_state_topic; a fresh state selects among these three.
        // heading_lookahead_m above is the startup default / fallback (no planner).
        look_straight_ = declare_parameter("heading_lookahead_straight_m", look_m);
        look_turn_ = declare_parameter("heading_lookahead_turn_m", look_m);
        look_winding_ = declare_parameter("heading_lookahead_winding_m", look_m);
        path_timeout_ = declare_parameter("path_timeout_sec", 2.0);
        stop_no_path_ = declare_parameter("stop_if_no_path", true);
        search_radius_ = declare_parameter("closest_point_search_radius_m", 2.0);
        goal_tol_ = declare_parameter("goal_tolerance_m", 0.35);

        std::string cmd_topic = declare_parameter("command_topic", std::string("/cmd_vel"));
        std::string odom_topic = declare_parameter("odom_topic", std::string("/odom_ekf"));
        std::string path_topic = declare_parameter("path_topic", std::string("/vlm_path_odom"));
        std::string ref_topic = declare_parameter("reference_path_topic", std::string("/mpc_reference_path"));

        // Runtime speed setpoint from the maneuver state machine (planner). target_speed_mps
        // above is only the STARTUP default/fallback; /maneuver/target_speed overrides it at
        // runtime (params are read once at startup — KB §2.16). A setpoint is only applied
        // while FRESH (received within speed_setpoint_timeout_sec); on a stale/missing read the
        // solver keeps its current target (fail-safe — never accelerate on a lost read).
        std::string spd_topic = declare_parameter("target_speed_topic", std::string("/maneuver/target_speed"));
        speed_setpoint_timeout_ = declare_parameter("speed_setpoint_timeout_sec", 1.0);

        // Slew-rate limits on the target-speed setpoint (v_ref), so a regime switch in the
        // maneuver FSM ramps v_ref instead of stepping it — keeping the QP's linearization
        // speed near the actual speed. Asymmetric ON PURPOSE: accel is slow (smooths the
        // turn->straight speed-up) while decel is fast, so a stop/slow-down setpoint is
        // never rate-limited into a safety problem.
        accel_rate_ = declare_parameter("target_speed_accel_rate_mps2", 0.8);
        decel_rate_ = declare_parameter("target_speed_decel_rate_mps2", 3.0);
        v_ref_cmd_ = t_spd; // start from the startup target
        std::string state_topic = declare_parameter("maneuver_state_topic", std::string("/maneuver/state"));

        MpcParams mp{};
        mp.N = N;
        mp.dt = dt;
        mp.target_speed = t_spd;
        mp.w_lat = w_lat;
        mp.w_head = w_hd;
        mp.w_speed = w_spd;
        mp.w_accel = w_acc;
        mp.w_steer = w_str;
        mp.w_accel_change = w_acc_rate;
        mp.w_steer_change = w_str_rate;
        mp.lookahead_m = look_m;
        VehicleParams vp{};
        vp.wheelbase = wb;
        vp.max_speed = max_spd;
        vp.max_accel = max_acc;
        vp.max_decel = max_dec;
        vp.max_steer = max_str;

        solver_ = std::make_unique<MpcSolver>(vp, mp);

        vp_ = vp;

        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            path_topic,
            10,
            [this](nav_msgs::msg::Path::SharedPtr m)
            { on_path(m); });

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic,
            10,
            [this](nav_msgs::msg::Odometry::SharedPtr m)
            { on_odom(m); });

        speed_sub_ = create_subscription<std_msgs::msg::Float64>(
            spd_topic,
            10,
            [this](std_msgs::msg::Float64::SharedPtr m)
            {
                speed_setpoint_ = m->data;
                speed_setpoint_time_ = now();
                got_setpoint_ = true;
            });

        state_sub_ = create_subscription<std_msgs::msg::String>(
            state_topic,
            10,
            [this](std_msgs::msg::String::SharedPtr m)
            {
                maneuver_state_ = m->data;
                maneuver_state_time_ = now();
                got_state_ = true;
            });

        cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(
            cmd_topic,
            10);

        pred_pub_ = create_publisher<nav_msgs::msg::Path>(
            "/mpc_predicted_path",
            10);

        ref_pub_ = create_publisher<nav_msgs::msg::Path>(
            ref_topic,
            10);

        // Control-debug readouts (consumed by the planner's FPV overlay and
        // recordable via debugkit): the slew-limited target speed the solver is
        // actually given, and the raw QP outputs (accel, steer).
        dbg_vref_pub_ = create_publisher<std_msgs::msg::Float64>("/mpc/v_ref_ramped", 10);
        dbg_accel_pub_ = create_publisher<std_msgs::msg::Float64>("/mpc/cmd_accel", 10);
        dbg_steer_pub_ = create_publisher<std_msgs::msg::Float64>("/mpc/cmd_steer", 10);

        auto period = std::chrono::duration<double>(1.0 / rate_hz_);

        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&MpcTrackerNode::control_tick, this)); // control loop timer

        RCLCPP_INFO(get_logger(), "MPC tracker started at %.1f Hz", rate_hz_);
    }

private:
    double rate_hz_;
    double path_timeout_, search_radius_, goal_tol_;
    bool stop_no_path_;
    bool arrived_ = false;
    double prev_accel_ = 0.0, prev_steer_ = 0.0;
    int last_source_ = -1; // -1 unknown, 0 = QP, 1 = proportional fallback
    std::unique_ptr<MpcSolver> solver_;

    // Runtime speed setpoint (/maneuver/target_speed). got_setpoint_ guards against the
    // stored 0.0 default reading as "fresh" before any message arrives (which under
    // use_sim_time can happen in the first second of sim time, when now() < timeout); the
    // solver then holds its startup target_speed_mps default until the first real setpoint.
    bool got_setpoint_ = false;
    double speed_setpoint_ = 0.0;
    double speed_setpoint_timeout_ = 1.0;
    rclcpp::Time speed_setpoint_time_{0, 0, RCL_ROS_TIME};

    // Slew-rate-limited target speed actually injected into the solver each tick.
    double v_ref_cmd_ = 0.0;
    double accel_rate_ = 0.8; // max v_ref rise rate [m/s^2]
    double decel_rate_ = 3.0; // max v_ref fall rate [m/s^2] (fast for safety)

    // Maneuver state (/maneuver/state) -> heading lookahead selection. Same
    // freshness pattern as the speed setpoint: applied only while fresh (shares
    // speed_setpoint_timeout_); stale/missing -> the solver keeps its current
    // lookahead (startup heading_lookahead_m until the first state arrives).
    bool got_state_ = false;
    std::string maneuver_state_;
    rclcpp::Time maneuver_state_time_{0, 0, RCL_ROS_TIME};
    double look_straight_ = 0.6, look_turn_ = 0.6, look_winding_ = 0.6;

    nav_msgs::msg::Path::SharedPtr path_;
    nav_msgs::msg::Odometry::SharedPtr odom_;
    rclcpp::Time path_time_{0, 0, RCL_ROS_TIME}; // time when the last path was received

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr speed_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pred_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ref_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr dbg_vref_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr dbg_accel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr dbg_steer_pub_;

    void publish_debug(double vref, double accel, double steer)
    {
        std_msgs::msg::Float64 m;
        m.data = vref;  dbg_vref_pub_->publish(m);
        m.data = accel; dbg_accel_pub_->publish(m);
        m.data = steer; dbg_steer_pub_->publish(m);
    }

    rclcpp::TimerBase::SharedPtr timer_;

    VehicleParams vp_;

    void on_path(nav_msgs::msg::Path::SharedPtr msg)
    {
        path_ = msg;
        path_time_ = now();
        arrived_ = false;
    }

    void on_odom(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        odom_ = msg;
    }

    void stop()
    {
        geometry_msgs::msg::Twist cmd{}; // zero command
        cmd_pub_->publish(cmd);          // publish zero command to stop the robot
        publish_debug(v_ref_cmd_, 0.0, 0.0); // overlay shows braking, not stale cmds
    }

    void control_tick()
    {
        if (!odom_)
            return;
        if (!path_ || path_->poses.empty())
        {
            if (stop_no_path_)
                stop();
            return;
        }

        double age = (now() - path_time_).seconds();
        if (age > path_timeout_)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Path stale (%.1fs), stopping",
                age);
            stop();
            return;
        }

        if (arrived_)
        {
            stop();
            return;
        }

        // odom ==> vehicle state
        VehicleState state;
        state.x = odom_->pose.pose.position.x;
        state.y = odom_->pose.pose.position.y;
        state.psi = tf2::getYaw(odom_->pose.pose.orientation);
        state.v = std::hypot(odom_->twist.twist.linear.x, odom_->twist.twist.linear.y);

        // reference path
        auto pts = path_msg_to_points(*path_);
        if (pts.empty())
        {
            stop();
            return;
        }

        const auto &goal_pose = path_->poses.back().pose.position;
        double dist_to_goal = std::hypot(goal_pose.x - state.x, goal_pose.y - state.y);
        if (dist_to_goal <= goal_tol_)
        {
            RCLCPP_INFO(get_logger(), "Reached end of path — stopping.");
            arrived_ = true;
            v_ref_cmd_ = 0.0; // a later re-plan ramps up from rest, not a jump
            stop();
            return;
        }

        size_t closest = find_closest_point(pts, state, search_radius_);

        // Resample N+1 points starting from closest
        double ds = state.v * 0.08 + 0.05; // adaptive spacing
        auto ref = resample_path(pts, closest, 16, std::max(ds, 0.05));

        // public reference path for visualization
        nav_msgs::msg::Path ref_msg;
        ref_msg.header.stamp = now();
        ref_msg.header.frame_id = "odom";
        for (const auto &p : ref)
        {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = ref_msg.header;
            ps.pose.position.x = p.x;
            ps.pose.position.y = p.y;
            ref_msg.poses.push_back(ps);
        }
        ref_pub_->publish(ref_msg);

        // Desired target speed: the maneuver setpoint iff fresh; else hold the current
        // ramped value (fail-safe — never accelerate on a stale/missing read).
        double desired = v_ref_cmd_;
        if (got_setpoint_ && (now() - speed_setpoint_time_).seconds() <= speed_setpoint_timeout_)
            desired = std::clamp(speed_setpoint_, 0.0, vp_.max_speed);

        // Slew v_ref toward the desired value (asymmetric: slow accel, fast decel) so a
        // regime switch ramps rather than steps — keeps the QP's linearization speed near
        // the actual speed. dt_tick is the control period.
        const double dt_tick = 1.0 / rate_hz_;
        double d = desired - v_ref_cmd_;
        double step = (d >= 0.0 ? accel_rate_ : decel_rate_) * dt_tick;
        v_ref_cmd_ += std::clamp(d, -step, step);
        solver_->set_target_speed(v_ref_cmd_);

        // Maneuver-dependent heading lookahead (see param comments): a fresh FSM
        // state selects among straight/turn/winding; stop rides the (short) turn
        // value — the halt approach is slow and its path is short. Stale/missing
        // state -> hold the solver's current lookahead.
        if (got_state_ && (now() - maneuver_state_time_).seconds() <= speed_setpoint_timeout_)
        {
            double look = look_straight_;
            if (maneuver_state_ == "left" || maneuver_state_ == "right" ||
                maneuver_state_ == "stop")
                look = look_turn_;
            else if (maneuver_state_ == "winding")
                look = look_winding_;
            solver_->set_lookahead(look);
        }

        // solve MPC
        auto result = solver_->Solve(state, ref, prev_accel_, prev_steer_);

        if (!result.success)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "MPC solve failed");
            stop();
            return;
        }

        // Announce the control source once per transition (QP <-> fallback), and
        // keep a throttled reminder while degraded to the proportional fallback.
        const int source = result.used_fallback ? 1 : 0;
        if (source != last_source_)
        {
            RCLCPP_INFO(get_logger(), "Control source: %s",
                        result.used_fallback ? "PROPORTIONAL fallback (QP failed)"
                                             : "MPC QP (OSQP)");
            last_source_ = source;
        }
        if (result.used_fallback)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "OSQP solve failing — driving on proportional fallback");
        }

        prev_accel_ = result.accel;
        prev_steer_ = result.steer;

        // convert steer to angular z
        double angular_z = state.v / vp_.wheelbase * std::tan(result.steer);
        double new_v = std::clamp(state.v + result.accel / rate_hz_, 0.0, vp_.max_speed);

        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = new_v;
        cmd.angular.z = angular_z;
        cmd_pub_->publish(cmd);
        publish_debug(v_ref_cmd_, result.accel, result.steer);

        nav_msgs::msg::Path pred;
        pred.header.stamp = now();
        pred.header.frame_id = "odom";
        for (const auto &s : result.predicted_states)
        {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = pred.header;
            ps.pose.position.x = s.x;
            ps.pose.position.y = s.y;
            pred.poses.push_back(ps);
        }
        pred_pub_->publish(pred);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MpcTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
