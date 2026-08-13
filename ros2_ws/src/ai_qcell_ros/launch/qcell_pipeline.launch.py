from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    model_path = LaunchConfiguration("model_path")
    dataset_root = LaunchConfiguration("dataset_root")
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path", description="Deep PatchCore .pt file"),
            DeclareLaunchArgument("dataset_root", description="MVTec bottle dataset root"),
            Node(
                package="ai_qcell_ros",
                executable="reject_action_server",
                name="reject_action_server",
                output="screen",
            ),
            Node(
                package="ai_qcell_ros",
                executable="inspection_node",
                name="inspection_node",
                output="screen",
                parameters=[{"model_path": model_path}],
            ),
            Node(
                package="ai_qcell_ros",
                executable="decision_node",
                name="decision_node",
                output="screen",
            ),
            Node(
                package="ai_qcell_ros",
                executable="dashboard_bridge",
                name="dashboard_bridge",
                output="screen",
            ),
            Node(
                package="ai_qcell_ros",
                executable="camera_node",
                name="camera_node",
                output="screen",
                parameters=[{"dataset_root": dataset_root, "publish_period": 2.0}],
            ),
        ]
    )
