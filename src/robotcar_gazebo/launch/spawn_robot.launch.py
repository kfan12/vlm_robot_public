from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Spawn pose — override at launch time, e.g.
    #   ros2 launch robotcar_gazebo spawn_robot.launch.py spawn_x:=0.0
    # or change the defaults here in one place.
    return LaunchDescription([
        DeclareLaunchArgument('spawn_x', default_value='-3.0',
                              description='Robot spawn X in world frame [m]'),
        DeclareLaunchArgument('spawn_y', default_value='0.0',
                              description='Robot spawn Y in world frame [m]'),
        DeclareLaunchArgument('spawn_z', default_value='0.12',
                              description='Robot spawn Z in world frame [m]'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0',
                              description='Robot spawn yaw [rad]'),

        # Delay 6s: Gazebo needs ~3s to load, bridge needs ~1s to connect,
        # robot_state_publisher needs /clock before publishing /robot_description.
        TimerAction(period=6.0, actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name',  'robotcar',
                    '-world', 'cone_lane',
                    '-topic', 'robot_description',
                    '-x', LaunchConfiguration('spawn_x'),
                    '-y', LaunchConfiguration('spawn_y'),
                    '-z', LaunchConfiguration('spawn_z'),
                    '-Y', LaunchConfiguration('spawn_yaw'),
                ],
                output='screen',
            ),
        ]),
    ])
