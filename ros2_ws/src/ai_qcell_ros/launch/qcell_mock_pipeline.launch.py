from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    dataset_root = LaunchConfiguration("dataset_root")
    publish_period = LaunchConfiguration("publish_period")
    return LaunchDescription(
        [
            DeclareLaunchArgument("dataset_root", description="MVTec bottle dataset root"),
            DeclareLaunchArgument("publish_period", default_value="1.0"),
            Node(
                package="ai_qcell_ros",
                executable="reject_action_server",
                output="screen",
            ),
            Node(
                package="ai_qcell_ros",
                executable="mock_inspection_node",
                output="screen",
            ),
            Node(package="ai_qcell_ros", executable="decision_node", output="screen"),
            Node(package="ai_qcell_ros", executable="dashboard_bridge", output="screen"),
            Node(
                package="ai_qcell_ros",
                executable="camera_node",
                output="screen",
                parameters=[
                    {"dataset_root": dataset_root, "publish_period": publish_period}
                ],
            ),
        ]
    )
