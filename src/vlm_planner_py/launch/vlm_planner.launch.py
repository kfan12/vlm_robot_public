import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('vlm_planner_py')
    default_params = os.path.join(pkg, 'config', 'vlm_params.yaml')
    return LaunchDescription([
        # Point this at the SOURCE yaml to tune per-sim WITHOUT a rebuild, e.g.
        #   params_file:=$HOME/vlm_robot_demo/src/vlm_planner_py/config/vlm_params.yaml
        # then just edit + relaunch this pane. Defaults to the installed copy.
        DeclareLaunchArgument('params_file', default_value=default_params),
        # Gate the planner's path output on the first /vlm/sign from the separate
        # vlm_sign node (set true when running alongside it). Default false = the
        # planner runs immediately/standalone. Overrides the yaml value.
        DeclareLaunchArgument('wait_for_first_sign', default_value='false'),
        Node(
            package='vlm_planner_py',
            executable='vlm_node',
            name='vlm_planner',
            parameters=[
                LaunchConfiguration('params_file'),
                {'wait_for_first_sign': ParameterValue(
                    LaunchConfiguration('wait_for_first_sign'), value_type=bool)},
            ],
            # A stray ~/.local numpy2 (pulled in by pandas, for unrelated Jupyter
            # tooling) shadows apt's numpy1 on system python's sys.path, which
            # crashes apt's python3-opencv (numpy1 ABI) on `import cv2`.
            # PYTHONNOUSERSITE skips user site-packages for just this process so
            # apt's numpy1 resolves again, without touching any installed package.
            additional_env={'PYTHONNOUSERSITE': '1'},
            output='screen',
        )
    ])
