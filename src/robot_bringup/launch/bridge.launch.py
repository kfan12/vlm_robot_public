import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_bringup = get_package_share_directory('robot_bringup')
    bridge_config = os.path.join(pkg_bringup, 'config', 'bridge.yaml')

    return LaunchDescription([
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_ros_bridge',
            parameters=[{'config_file': bridge_config}],
            output='screen',
        ),
    ])
