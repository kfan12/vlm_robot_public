import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Render a ghost of the robot at its ground-truth sim pose in RViz:
       odom_tf_broadcaster  : odom -> truth/base_link from /odom_truth
       robot_state_publisher: the truth/* link frames (frame_prefix 'truth/')
       (RViz then shows a second RobotModel with TF Prefix 'truth')."""
    urdf = os.path.join(
        get_package_share_directory('robotcar_description'),
        'urdf', 'robotcar.urdf.xacro')
    robot_description = ParameterValue(Command(['xacro ', urdf]), value_type=str)

    return LaunchDescription([
        Node(
            package='vlm_planner_py',
            executable='odom_tf_broadcaster',
            name='odom_tf_broadcaster',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'odom_topic': '/odom_truth',
                'child_frame': 'truth/base_link',
            }],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='rsp_truth',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
                'frame_prefix': 'truth/',
            }],
            # Don't clobber the main /robot_description latched topic.
            remappings=[('/robot_description', '/robot_description_truth')],
        ),
    ])
