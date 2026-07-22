import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_robotcar_desc = get_package_share_directory('robotcar_description')
    pkg_robotcar_gz   = get_package_share_directory('robotcar_gazebo')
    pkg_bringup       = get_package_share_directory('robot_bringup')

    world_file   = os.path.join(pkg_robotcar_gz,   'worlds', 'cone_lane.world.sdf')
    urdf_file    = os.path.join(pkg_robotcar_desc,  'urdf',   'robotcar.urdf.xacro')
    bridge_cfg   = os.path.join(pkg_bringup,        'config', 'bridge.yaml')

    robot_description = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)

    return LaunchDescription([

        # 1. Gazebo — starts immediately
        ExecuteProcess(
            cmd=['ign', 'gazebo', '-r', world_file],
            additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'},
            output='screen'
        ),

        # 2. robot_state_publisher — starts immediately, waits for /clock
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }]
        ),

        # 3. Bridge — delayed 3s to let Gazebo fully load before connecting
        TimerAction(period=3.0, actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='gz_ros_bridge',
                parameters=[{'config_file': bridge_cfg}],
                output='screen',
            ),
        ]),


        # 4. path_marker_node — draws the planned /vlm_path_truth (ground-truth frame)
        #    into the Gazebo scene as a GREEN ribbon. Delayed 8s so Gazebo (and
        #    its marker service) is up.
        TimerAction(period=8.0, actions=[
            Node(
                package='robotcar_utils_cpp',
                executable='path_marker_node',
                name='path_marker_node',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'path_topic': '/vlm_path_truth',    # vlm_path_truth based on ground truth odom
                    'odom_topic': '/odom_truth',
                    'marker_ns': 'vlm_path_truth',
                    'marker_id': 1,
                    'color_r': 0.0, 'color_g': 1.0, 'color_b': 0.0,  # green
                }],
            ),

            # 5. Second instance — the controller's reference path /vlm_path_odom
            #    (wheel frame) as a RED ribbon, mapped to world via wheel /odom.
            #    track_continuously so it follows odometry drift (diverges from
            #    green as drift accumulates).
            # Node(
            #     package='robotcar_utils_cpp',
            #     executable='path_marker_node',
            #     name='path_marker_ctrl',
            #     output='screen',
            #     parameters=[{
            #         'use_sim_time': True,
            #         'path_topic': '/vlm_path_odom',
            #         'odom_topic': '/odom',
            #         'marker_ns': 'vlm_path_ctrl',
            #         'marker_id': 2,
            #         'track_continuously': True,
            #         'color_r': 1.0, 'color_g': 0.0, 'color_b': 0.0,  # red
            #     }],
            # ),
        ]),
    ])