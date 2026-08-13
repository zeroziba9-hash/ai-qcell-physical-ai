import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ai_qcell_interfaces.msg import InspectionResult


class DashboardBridge(Node):
    def __init__(self) -> None:
        super().__init__("dashboard_bridge")
        self.create_subscription(
            InspectionResult, "/qcell/inspection/result", self.on_inspection, 10
        )
        self.create_subscription(String, "/qcell/sort/pass", self.on_pass, 10)
        self.create_subscription(String, "/qcell/reject/status", self.on_reject, 10)

    def on_inspection(self, message: InspectionResult) -> None:
        payload = {
            "type": "inspection",
            "product_id": message.product_id,
            "is_defect": message.is_defect,
            "raw_score": round(message.raw_score, 6),
            "latency_ms": round(message.latency_ms, 1),
        }
        self.get_logger().info(json.dumps(payload, ensure_ascii=False))

    def on_pass(self, message: String) -> None:
        self.get_logger().info(json.dumps({"type": "sort", "value": message.data}))

    def on_reject(self, message: String) -> None:
        self.get_logger().info(json.dumps({"type": "reject", "value": message.data}))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DashboardBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
