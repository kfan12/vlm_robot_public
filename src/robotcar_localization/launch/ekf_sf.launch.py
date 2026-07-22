import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Hand-written C++ EKF: fuses /odom + /imu into /odom_sf (no TF by default)."""
    cfg = os.path.join(
        get_package_share_directory('robotcar_localization'), 'config', 'ekf_sf.yaml')

    return LaunchDescription([
        Node(
            package='robotcar_localization',
            executable='ekf_sf_node',
            name='ekf_sf_node',
            output='screen',
            parameters=[cfg, {'use_sim_time': True}],
        ),
    ])
