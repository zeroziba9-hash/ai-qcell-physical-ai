from glob import glob
from setuptools import find_packages, setup


package_name = "ai_qcell_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AI-QCell",
    maintainer_email="portfolio@example.com",
    description="AI-QCell ROS2 inspection and sorting nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = ai_qcell_ros.camera_node:main",
            "inspection_node = ai_qcell_ros.inspection_node:main",
            "mock_inspection_node = ai_qcell_ros.mock_inspection_node:main",
            "decision_node = ai_qcell_ros.decision_node:main",
            "reject_action_server = ai_qcell_ros.reject_action_server:main",
            "dashboard_bridge = ai_qcell_ros.dashboard_bridge:main",
        ],
    },
)
