import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Localization stack:
       - robot_localization ekf_node : fuses /odom + /imu -> /odom_ekf (owns odom->base_link TF)

       The hand-written comparison EKF (ekf_sf_node -> /odom_sf) is disabled below:
       nothing in the control chain consumes /odom_sf, and the duplicate fusion
       work contributed to ekf_node missing its update rate on the loaded WSL2
       CPU. Uncomment to re-run the Day-13 comparison.
    """
    pkg = get_package_share_directory('robotcar_localization')
    ekf_cfg = os.path.join(pkg, 'config', 'ekf.yaml')
    # ekf_sf_cfg = os.path.join(pkg, 'config', 'ekf_sf.yaml')

    return LaunchDescription([
        Node(
            package='robot_localization',  # official ROS2 package
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_cfg, {'use_sim_time': True}],
            remappings=[('odometry/filtered', '/odom_ekf')],
        ),
        # Node(
        #     package='robotcar_localization', # hand-written C++ EKF for comparison
        #     executable='ekf_sf_node',
        #     name='ekf_sf_node',
        #     output='screen',
        #     parameters=[ekf_sf_cfg, {'use_sim_time': True}],
        # ),
    ])
