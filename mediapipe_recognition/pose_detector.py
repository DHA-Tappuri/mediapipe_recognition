import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

class PoseDetectorNode(Node):
    def __init__(self):
        super().__init__('pose_detector_node')
        self._sub_image  = self.create_subscription(Image, '/image_raw', self._callback_image, 10)
        self._pub_result = self.create_publisher(Image, '/image_result2', 10)
        self._bridge     = CvBridge()

        self._mp_pose    = mp.solutions.pose
        self._pose       = self._mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self._mp_drawing = mp.solutions.drawing_utils


    def _callback_image(self, msg):
        _frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        _rgb = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
        _res = self._pose.process(_rgb)
        
        if(_res.pose_landmarks):
            self._mp_drawing.draw_landmarks(_frame, _res.pose_landmarks, self._mp_pose.POSE_CONNECTIONS)
            
        _result = self._bridge.cv2_to_imgmsg(_frame, encoding='bgr8')
        self._pub_result.publish(_result)



# main function
def main(args=None):
    rclpy.init(args=args)
    node = PoseDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# entry point
if(__name__ == '__main__'):
    main()

