import rclpy
from rclpy.node import Node

from ai_qcell_interfaces.msg import InspectionResult, ProductFrame


class MockInspectionNode(Node):
    """Deterministic inference substitute for validating the real ROS2 graph."""

    def __init__(self) -> None:
        super().__init__("mock_inspection_node")
        self.publisher = self.create_publisher(
            InspectionResult, "/qcell/inspection/result", 10
        )
        self.subscription = self.create_subscription(
            ProductFrame, "/qcell/camera/product", self.inspect, 10
        )
        self.get_logger().info("mock inspection ready for ROS2 integration test")

    def inspect(self, frame: ProductFrame) -> None:
        is_defect = frame.defect_type != "good"
        result = InspectionResult()
        result.stamp = self.get_clock().now().to_msg()
        result.product_id = frame.product_id
        result.image_path = frame.image_path
        result.defect_type = frame.defect_type
        result.is_defect = is_defect
        result.anomaly_score = 78.0 if is_defect else 28.0
        result.raw_score = 0.72 if is_defect else 0.26
        result.threshold = 0.40
        result.latency_ms = 8.0
        self.publisher.publish(result)
        self.get_logger().info(
            f"{result.product_id}: {'REJECT' if is_defect else 'PASS'} "
            f"type={result.defect_type}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockInspectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
