import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from ai_qcell_interfaces.action import RejectProduct
from ai_qcell_interfaces.msg import InspectionResult


class DecisionNode(Node):
    def __init__(self) -> None:
        super().__init__("decision_node")
        self.action_client = ActionClient(self, RejectProduct, "/qcell/reject_product")
        self.pass_publisher = self.create_publisher(String, "/qcell/sort/pass", 10)
        self.status_publisher = self.create_publisher(String, "/qcell/reject/status", 10)
        self.subscription = self.create_subscription(
            InspectionResult, "/qcell/inspection/result", self.decide, 10
        )

    def decide(self, result: InspectionResult) -> None:
        if not result.is_defect:
            message = String()
            message.data = f"{result.product_id}|PASS_LANE"
            self.pass_publisher.publish(message)
            self.get_logger().info(message.data)
            return

        if not self.action_client.server_is_ready():
            self.get_logger().error("reject action server is not ready")
            return
        goal = RejectProduct.Goal()
        goal.product_id = result.product_id
        goal.reason = (
            f"raw_score={result.raw_score:.4f} exceeded threshold={result.threshold:.4f}"
        )
        future = self.action_client.send_goal_async(goal, feedback_callback=self.on_feedback)
        future.add_done_callback(self.on_goal_response)

    def on_feedback(self, feedback_message) -> None:
        feedback = feedback_message.feedback
        self.get_logger().info(f"reject {feedback.progress:.0f}%: {feedback.state}")

    def on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("reject goal was refused")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_result(self, future) -> None:
        result = future.result().result
        message = String()
        message.data = f"success={result.success}|{result.final_state}"
        self.status_publisher.publish(message)
        self.get_logger().info(message.data)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
