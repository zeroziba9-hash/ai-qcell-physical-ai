from pathlib import Path

import rclpy
from rclpy.node import Node

from ai_qcell_interfaces.msg import ProductFrame


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")
        self.declare_parameter("dataset_root", "")
        self.declare_parameter("publish_period", 2.0)
        dataset_root = Path(str(self.get_parameter("dataset_root").value))
        self.samples = sorted((dataset_root / "test").glob("*/*.png"))
        self.publisher = self.create_publisher(ProductFrame, "/qcell/camera/product", 10)
        self.index = 0
        period = float(self.get_parameter("publish_period").value)
        self.timer = self.create_timer(period, self.publish_next)
        if self.samples:
            self.get_logger().info(f"camera ready: {len(self.samples)} MVTec samples")
        else:
            self.get_logger().error(f"no test images under {dataset_root / 'test'}")

    def publish_next(self) -> None:
        if not self.samples:
            return
        path = self.samples[self.index % len(self.samples)]
        self.index += 1
        message = ProductFrame()
        message.stamp = self.get_clock().now().to_msg()
        message.product_id = f"QCELL-{self.index:04d}"
        message.image_path = str(path)
        message.defect_type = path.parent.name
        self.publisher.publish(message)
        self.get_logger().info(f"published {message.product_id}: {path.name}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
