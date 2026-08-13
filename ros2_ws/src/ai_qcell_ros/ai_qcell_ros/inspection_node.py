from pathlib import Path

from PIL import Image
import rclpy
from rclpy.node import Node

from ai_qcell_interfaces.msg import InspectionResult, ProductFrame
from qcell.deep_patchcore import DeepPatchCore


class InspectionNode(Node):
    def __init__(self) -> None:
        super().__init__("inspection_node")
        self.declare_parameter("model_path", "")
        model_path = Path(str(self.get_parameter("model_path").value))
        if not model_path.is_file():
            raise FileNotFoundError(f"Deep PatchCore model not found: {model_path}")
        self.model = DeepPatchCore.load(model_path)
        self.publisher = self.create_publisher(
            InspectionResult, "/qcell/inspection/result", 10
        )
        self.subscription = self.create_subscription(
            ProductFrame, "/qcell/camera/product", self.inspect, 10
        )
        self.get_logger().info(f"Deep PatchCore loaded: {model_path}")

    def inspect(self, frame: ProductFrame) -> None:
        try:
            prediction = self.model.predict(Image.open(frame.image_path).convert("RGB"))
        except Exception as error:
            self.get_logger().error(f"inspection failed for {frame.product_id}: {error}")
            return
        result = InspectionResult()
        result.stamp = self.get_clock().now().to_msg()
        result.product_id = frame.product_id
        result.image_path = frame.image_path
        result.defect_type = frame.defect_type
        result.is_defect = prediction.is_defect
        result.anomaly_score = prediction.anomaly_score
        result.raw_score = prediction.raw_score
        result.threshold = prediction.threshold
        result.latency_ms = prediction.latency_ms
        self.publisher.publish(result)
        decision = "REJECT" if result.is_defect else "PASS"
        self.get_logger().info(
            f"{result.product_id}: {decision} score={result.raw_score:.4f} "
            f"latency={result.latency_ms:.1f}ms"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InspectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
