"""LIMO (four-diff) MPPI launch in the Gazebo corridor-pedestrian scene.

Ported from the mosaic_mppi_jax workspace to the EXACT exact_mppi_jax bridge.
Spawns the AgileX LIMO four-diff URDF (from limo_description) into a randomized
corridor-pedestrian world and runs the installed `exact_mppi` JAX controller
through the ROS2 bridge package.

Starts:
    - Gazebo Classic with a generated dynamic-pedestrian corridor world
    - LIMO URDF spawned via xacro -> robot_description -> spawn_entity.py
    - robot_state_publisher (TF)
    - global_ref_path + mppi_local (exact_mppi_jax)
    - map -> odom static TF
    - gazebo_dynamic_obstacles driver
    - corridor_goal_trigger (optional, auto_goal:=true)
    - optional RViz

Usage:
  ros2 launch exact_mppi_jax sim_limo_corridor_launch.py
  ros2 launch exact_mppi_jax sim_limo_corridor_launch.py use_rviz:=false
  ros2 launch exact_mppi_jax sim_limo_corridor_launch.py auto_goal:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.logging import get_logger
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from exact_mppi_jax.corridor_pedestrian_scene import generate_corridor_pedestrian_scene


logger = get_logger("exact_mppi_jax_sim_limo_corridor_launch")

MPPI_CONFIG_FILE = "limo_corridor_planner.yaml"


def generate_launch_description() -> LaunchDescription:
    pkg_dir = get_package_share_directory("exact_mppi_jax")
    rviz_config = os.path.join(pkg_dir, "rviz", "sim_limo_corridor.rviz")
    pedestrian_model_path = os.path.join(pkg_dir, "models")

    limo_description_dir = get_package_share_directory("limo_description")
    limo_xacro_path = os.path.join(limo_description_dir, "urdf", "limo_four_diff.xacro")

    x_pose_arg = DeclareLaunchArgument("x_pose", default_value="0.0")
    y_pose_arg = DeclareLaunchArgument("y_pose", default_value="9.5")
    yaw_arg = DeclareLaunchArgument("yaw", default_value="0.0")
    use_rviz_arg = DeclareLaunchArgument("use_rviz", default_value="true")
    ideal_gazebo_arg = DeclareLaunchArgument(
        "ideal_gazebo",
        default_value="true",
        description="Use a more kinematic skid-steer Gazebo drive plugin for the LIMO robot",
    )
    gui_arg = DeclareLaunchArgument("gui", default_value="true")
    paused_arg = DeclareLaunchArgument("paused", default_value="false")
    dynamic_obstacles_arg = DeclareLaunchArgument("dynamic_obstacles", default_value="true")
    auto_goal_arg = DeclareLaunchArgument(
        "auto_goal",
        default_value="false",
        description="Automatically publish the fixed corridor goal on startup. "
        "When false, trigger the goal manually (e.g. RViz 2D Goal Pose -> /goal_pose).",
    )
    obstacle_seed_arg = DeclareLaunchArgument(
        "obstacle_seed",
        default_value="",
        description="Optional integer seed for reproducible randomized pedestrian layout",
    )

    gazebo_model_path = SetEnvironmentVariable(
        "GAZEBO_MODEL_PATH",
        pedestrian_model_path
        + os.pathsep
        + os.environ.get("GAZEBO_MODEL_PATH", ""),
    )
    jax_no_prealloc = SetEnvironmentVariable("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    jax_mem_fraction = SetEnvironmentVariable("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

    def launch_generated_scene(context):
        scene = generate_corridor_pedestrian_scene(
            pkg_dir,
            LaunchConfiguration("obstacle_seed").perform(context),
        )
        logger.info(
            "Generated randomized corridor pedestrians "
            f"(seed={scene.seed}, world={scene.world_path}, config={scene.dynamic_config_path})"
        )
        gzserver_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([FindPackageShare("gazebo_ros"), "/launch/gzserver.launch.py"]),
            launch_arguments={
                "world": scene.world_path,
                "verbose": "false",
                "pause": LaunchConfiguration("paused"),
            }.items(),
        )
        dynamic_obstacles_node = Node(
            package="exact_mppi_jax",
            executable="gazebo_dynamic_obstacles",
            name="gazebo_dynamic_obstacles",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"use_sim_time": True},
                {"config_file": scene.dynamic_config_path},
            ],
            condition=IfCondition(LaunchConfiguration("dynamic_obstacles")),
        )
        return [gzserver_launch, dynamic_obstacles_node]

    gzclient_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("gazebo_ros"), "/launch/gzclient.launch.py"]),
        launch_arguments={"verbose": "false"}.items(),
        condition=IfCondition(LaunchConfiguration("gui")),
    )

    robot_description_cmd = Command([
        "xacro ",
        limo_xacro_path,
        " ideal_gazebo:=",
        LaunchConfiguration("ideal_gazebo"),
    ])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "robot_description": ParameterValue(robot_description_cmd, value_type=str),
        }],
    )

    # Publishes the 4 continuous wheel joints (zeroed) so robot_state_publisher
    # always emits base_link -> *_wheel_link TF, independent of the Gazebo
    # diff-drive plugin timing. Purely cosmetic (silences RViz RobotModel
    # "No transform to map" warnings for the wheels).
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "robot_description": ParameterValue(robot_description_cmd, value_type=str),
        }],
    )

    spawn_limo = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_limo",
        output="screen",
        arguments=[
            "-entity", "limo",
            "-topic", "robot_description",
            "-x", LaunchConfiguration("x_pose"),
            "-y", LaunchConfiguration("y_pose"),
            "-z", "0.05",
            "-Y", LaunchConfiguration("yaw"),
        ],
    )

    static_map_odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_map_odom_tf",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{"use_sim_time": True}],
    )

    global_ref_node = Node(
        package="exact_mppi_jax",
        executable="global_ref_path",
        name="global_ref_path",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"use_sim_time": True},
            {"mppi_config_file": MPPI_CONFIG_FILE},
        ],
    )

    mppi_local_node = Node(
        package="exact_mppi_jax",
        executable="mppi_local",
        name="mppi_local",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"use_sim_time": True},
            {"mppi_config_file": MPPI_CONFIG_FILE},
        ],
        remappings=[
            ("/mppi_cmd_vel", "/cmd_vel"),
        ],
    )

    corridor_goal_trigger = Node(
        package="exact_mppi_jax",
        executable="corridor_goal_trigger",
        name="corridor_goal_trigger",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"goal_topic": "/goal_pose"},
            {"frame_id": "map"},
            {"goal_x": 25.0},
            {"goal_y": 9.5},
            {"goal_yaw": 0.0},
            {"initial_delay": 2.0},
            {"publish_count": 10},
            {"publish_period": 0.2},
        ],
        condition=IfCondition(LaunchConfiguration("auto_goal")),
    )

    start_obstacles_on_goal = Node(
        package="exact_mppi_jax",
        executable="start_obstacles_on_goal",
        name="start_obstacles_on_goal",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"use_sim_time": True},
            {"goal_topic": "/goal_pose"},
            {"start_topic": "/start_dynamic_obstacles"},
        ],
        condition=IfCondition(LaunchConfiguration("dynamic_obstacles")),
    )

    rviz_args = ["-d", rviz_config] if os.path.exists(rviz_config) else []
    if not rviz_args:
        logger.warning(f"RViz config not found: {rviz_config}")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=rviz_args,
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription([
        x_pose_arg,
        y_pose_arg,
        yaw_arg,
        use_rviz_arg,
        ideal_gazebo_arg,
        gui_arg,
        paused_arg,
        dynamic_obstacles_arg,
        auto_goal_arg,
        obstacle_seed_arg,
        gazebo_model_path,
        jax_no_prealloc,
        jax_mem_fraction,
        OpaqueFunction(function=launch_generated_scene),
        gzclient_launch,
        robot_state_publisher,
        joint_state_publisher,
        spawn_limo,
        static_map_odom_tf,
        global_ref_node,
        mppi_local_node,
        corridor_goal_trigger,
        start_obstacles_on_goal,
        rviz_node,
    ])
