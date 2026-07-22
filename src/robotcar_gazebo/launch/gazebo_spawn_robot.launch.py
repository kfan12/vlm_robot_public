import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_robotcar_desc = get_package_share_directory('robotcar_description')
    pkg_robotcar_gz   = get_package_share_directory('robotcar_gazebo')
    pkg_bringup       = get_package_share_directory('robot_bringup')

    # World SDF — override at launch time, e.g. world:=curved_cone_lane.world.sdf.
    # All world SDFs share the world NAME 'cone_lane', so the spawn's
    # -world argument below stays unchanged regardless of which file is loaded.
    declare_world = DeclareLaunchArgument('world', default_value='track_lines.world.sdf',
                                          description='World SDF filename under robotcar_gazebo/worlds')
    world_file   = PathJoinSubstitution([pkg_robotcar_gz, 'worlds', LaunchConfiguration('world')])
    urdf_file    = os.path.join(pkg_robotcar_desc,  'urdf',   'robotcar.urdf.xacro')
    bridge_cfg   = os.path.join(pkg_bringup,        'config', 'bridge.yaml')

    robot_description = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)

    # Resource path so `model://<name>` in the world resolves to models under
    # robotcar_gazebo/models/. Must point at the immediate parent of each
    # model.config dir. Two roots: models/signs (sign_* boards) and models/
    # itself (ground_asphalt, added 2026-07-18) -- both prepended, preserving
    # any value already set in the environment.
    signs_models_dir = os.path.join(pkg_robotcar_gz, 'models', 'signs')
    all_models_dir = os.path.join(pkg_robotcar_gz, 'models')
    ign_resource_path = (signs_models_dir + os.pathsep + all_models_dir + os.pathsep
                         + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', ''))

    # Spawn pose — override at launch time, e.g.
    #   ros2 launch robotcar_gazebo gazebo_spawn_robot.launch.py spawn_x:=0.0
    # or just change the defaults here in one place.
    declare_spawn_x = DeclareLaunchArgument('spawn_x', default_value='-1.5',
                                            description='Robot spawn X in world frame [m]')
    declare_spawn_y = DeclareLaunchArgument('spawn_y', default_value='0.0',
                                            description='Robot spawn Y in world frame [m]')
    declare_spawn_z = DeclareLaunchArgument('spawn_z', default_value='0.12',
                                            description='Robot spawn Z in world frame [m]')
    declare_spawn_yaw = DeclareLaunchArgument('spawn_yaw', default_value='0.0',
                                              description='Robot spawn yaw [rad]')
    # headless:=true runs the Gazebo SERVER only (no GUI window). Sensor
    # rendering (rgbd camera) happens server-side either way; --headless-rendering
    # keeps it working without a usable display (autotune loops, CI).
    declare_headless = DeclareLaunchArgument('headless', default_value='false',
                                             description='Run Gazebo server-only (no GUI)')

    return LaunchDescription([
        declare_world,
        declare_spawn_x,
        declare_spawn_y,
        declare_spawn_z,
        declare_spawn_yaw,
        declare_headless,

        # 1. Gazebo — starts immediately (GUI by default, server-only if headless)
        ExecuteProcess(
            condition=UnlessCondition(LaunchConfiguration('headless')),
            cmd=['ign', 'gazebo', '-r', world_file],
            additional_env={
                'LIBGL_ALWAYS_SOFTWARE': '1',
                'IGN_GAZEBO_RESOURCE_PATH': ign_resource_path,
            },
            output='screen'
        ),
        # NOTE: no --headless-rendering. On WSL2 software GL that EGL path
        # DEGRADES then FREEZES the camera sensor after ~40-90 s (frames go
        # dark, then repeat bit-identically — autotune loops 1/3/4/5, see
        # docs/autotune_log.md). Server-only + WSLg's X (GLX) renders sensors
        # offscreen through the same path as the validated GUI runs.
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('headless')),
            cmd=['ign', 'gazebo', '-s', '-r', world_file],
            additional_env={
                'LIBGL_ALWAYS_SOFTWARE': '1',
                'IGN_GAZEBO_RESOURCE_PATH': ign_resource_path,
            },
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
                parameters=[{
                    'config_file': bridge_cfg,                 
                    }],
                output='screen',
            ),
        ]),


        # 3b. Spawn the robot in gazebo
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

        # cone_markers — publishes the static odom->world TF (needed by
        # fpv_truth_overlay). World-object MARKERS are turned OFF so RViz no longer
        # shows the world cones/objects; the TF is still published.
        Node(
            package='robotcar_utils_py',
            executable='cone_markers',
            name='cone_marker_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'publish_markers': False,   # no world objects in RViz; keep the TF
                'world_file': ParameterValue(world_file, value_type=str),
                'spawn_x': ParameterValue(LaunchConfiguration('spawn_x'), value_type=float),
                'spawn_y': ParameterValue(LaunchConfiguration('spawn_y'), value_type=float),
                'spawn_yaw': ParameterValue(LaunchConfiguration('spawn_yaw'), value_type=float),
            }],
        ),

        # odom_truth_rezero to shift odom truth coordinates to robot origin.
        # A SINGLE node re-zeros the odom (one in -> one out) AND every
        # ground-truth-anchored path via parallel path_in/out lists. Running two
        # instances here would duplicate the /odom_truth_rezero publisher and
        # share one node name — hence one node, many paths.
        Node(
            package='robotcar_utils_py',
            executable='odom_path_rezero',
            name='odom_path_rezero',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'odom_in_topic': '/odom_truth',
                'odom_out_topic': '/odom_truth_rezero',
                'path_in_topics': ['/vlm_path_truth', '/mpc_reference_path_truth'],
                'path_out_topics': ['/vlm_path_truth_rezero', '/mpc_reference_path_truth_rezero'],
            }],
        ),

        # fpv_truth_overlay — re-anchor the camera image to a ground-truth optical
        # frame so the RViz FPV overlay tracks the true pose (not the drifting
        # EKF). Republishes /fpv_truth/image_raw + /fpv_truth/camera_info.
        # image_in is the DETECTION overlay /vlm/debug_image (the raw frame with
        # the cone/goal/line detections drawn on it) instead of the raw camera, so
        # the single FPV Camera overlay in RViz shows the detections locked onto
        # the 3D scene — merging the old flat "Detection Overlay" Image display
        # into this one overlay. NOTE: /vlm/debug_image is published at planner
        # rate (and only while the planner runs), so the overlay refreshes slower
        # than the raw camera and goes blank if the planner is down.
        Node(
            package='robotcar_utils_py',
            executable='fpv_truth_overlay',
            name='fpv_truth_overlay',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'truth_topic': '/odom_truth',
                'image_in': '/camera/front/image_raw', #'/vlm/debug_image',
            }],
        ),


        # path_relay_node — re-express the PUBLISHED planner paths /vlm_path_odom
        # and /mpc_reference_path (anchored in the EKF odom frame) into the
        # ground-truth frame as /vlm_path_truth and /mpc_reference_truth, so they
        # can be drawn in Gazebo against the TRUE robot. The relay maps
        # "source-odom frame" -> "dest-odom frame": here the source is the EKF
        # pose the paths were built from (odom_src_topic := /odom_ekf) and the
        # destination is ground truth (odom_dest_topic := /odom_truth). One node
        # relays BOTH paths via parallel path_in/out lists, each keeping its own
        # locked transform and republish state.
        Node(
            package='mpc_tracker_cpp',
            executable='path_relay_node',
            name='path_relay_truth',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'path_in_topics': ['/vlm_path_odom', '/mpc_reference_path'],
                'path_out_topics': ['/vlm_path_truth', '/mpc_reference_path_truth'],
                'odom_src_topic': '/odom_ekf',    # SOURCE: odom frame the paths are anchored in
                'odom_dest_topic': '/odom_truth',  # DEST: odom frame to re-express the paths into
                'out_frame_id': 'odom',
            }],
        ),


        # 4. path_marker_node — draws the reference path /vlm_path_truth (now in the
        #    ground-truth frame, fed by path_relay_truth above) into the Gazebo
        #    scene as a GREEN ribbon, anchored via the constant /odom_truth->world
        #    transform. Delayed 8s so Gazebo (and its marker service) is up.
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
                    'line_width': 0.02,                              # ribbon width [m]
                    'color_r': 0.0, 'color_g': 1.0, 'color_b': 0.0,  # green
                }],
            ),

            Node(
                package='robotcar_utils_cpp',
                executable='path_marker_node',
                name='mpc_reference_marker_node',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    # Must match path_relay_truth's output topic; the relay now
                    # publishes /mpc_reference_path_truth (was /mpc_reference_truth).
                    'path_topic': '/mpc_reference_path_truth',    # ground-truth frame
                    'odom_topic': '/odom_truth',
                    'marker_ns': 'mpc_reference_path_truth',
                    'marker_id': 3,
                    'line_width': 0.05,                              # ribbon width [m]
                    'color_r': 1.0, 'color_g': 0.5, 'color_b': 0.0,  # yellow
                }],
            ),

        

            # 5. Second instance — the PUBLISHED planner path /vlm_path_odom as a
            #    RED ribbon. It is anchored in the EKF odom frame (the planner
            #    builds it from its /odom_ekf pose), so the odom->world transform
            #    MUST use /odom_ekf too, else the ribbon is offset. marker_id=2 and
            #    a distinct ns keep it separate from the green ground-truth ribbon.
            #    track_continuously so it follows EKF drift.
            # Node(
            #     package='robotcar_utils_cpp',
            #     executable='path_marker_node',
            #     name='path_marker_ctrl',
            #     output='screen',
            #     parameters=[{
            #         'use_sim_time': True,
            #         'path_topic': '/vlm_path_odom',
            #         'odom_topic': '/odom_ekf',
            #         'marker_ns': 'vlm_path_ctrl',
            #         'marker_id': 2,
            #         'track_continuously': True,
            #         'color_r': 1.0, 'color_g': 0.0, 'color_b': 0.0,  # red
            #     }],
            # ),
        ]),
    ])