import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('mpc_tracker_cpp')
    default_params = os.path.join(pkg, 'config', 'mpc_params.yaml')

    # params_file is overridable at launch time so the MPC can be retuned WITHOUT
    # a rebuild: point it at any editable yaml (e.g. the one in the source tree)
    # and just relaunch the node. Defaults to the installed config so the package
    # still works standalone.
    #   ros2 launch mpc_tracker_cpp mpc_tracker.launch.py \
    #     params_file:=$HOME/vlm_robot_demo/src/mpc_tracker_cpp/config/mpc_params.yaml
    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the mpc_tracker params yaml (edit + relaunch, no rebuild).')

    return LaunchDescription([
        declare_params,
        Node(
            package='mpc_tracker_cpp',
            executable='mpc_tracker_node',
            name='mpc_tracker',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
        ),
    ])
