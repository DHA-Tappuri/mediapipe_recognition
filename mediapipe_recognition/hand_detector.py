import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

class HandDetectorNode(Node):
    def __init__(self):
        super().__init__('hand_detector_node')
        self._sub_image  = self.create_subscription(Image, '/image_raw', self._callback_image, 10)
        self._pub_result = self.create_publisher(Image, '/image_result', 10)
        self._bridge     = CvBridge()

        self._mp_hands   = mp.solutions.hands
        self._hands      = self._mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self._mp_drawing = mp.solutions.drawing_utils


    def _callback_image(self, msg):
        _frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        _rgb = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
        _res = self._hands.process(_rgb)
        
        if(_res.multi_hand_landmarks):
            for _lm in _res.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(_frame, _lm, self._mp_hands.HAND_CONNECTIONS)

        _result = self._bridge.cv2_to_imgmsg(_frame, encoding='bgr8')
        self._pub_result.publish(_result)



# main function
def main(args=None):
    rclpy.init(args=args)
    node = HandDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# entry point
if(__name__ == '__main__'):
    main()

