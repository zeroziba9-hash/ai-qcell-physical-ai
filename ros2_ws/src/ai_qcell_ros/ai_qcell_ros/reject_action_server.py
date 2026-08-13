from time import sleep

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from ai_qcell_interfaces.action import RejectProduct


class RejectActionServer(Node):
    def __init__(self) -> None:
        super().__init__("reject_action_server")
        self.server = ActionServer(
            self,
            RejectProduct,
            "/qcell/reject_product",
            execute_callback=self.execute,
        )

    def execute(self, goal_handle):
        product_id = goal_handle.request.product_id
        self.get_logger().info(f"reject goal accepted: {product_id}")
        feedback = RejectProduct.Feedback()
        for progress, state in (
            (25.0, "gate preparing"),
            (50.0, "gate extending"),
            (75.0, "product diverting"),
            (100.0, "gate retracted"),
        ):
            feedback.progress = progress
            feedback.state = state
            goal_handle.publish_feedback(feedback)
            sleep(0.12)
        goal_handle.succeed()
        result = RejectProduct.Result()
        result.success = True
        result.final_state = f"{product_id}|REJECT_BIN"
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RejectActionServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
