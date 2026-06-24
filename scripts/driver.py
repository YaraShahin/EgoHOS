"""EgoHOS perception node.

Subscribes to an RGB image topic and runs the EgoHOS hand / first-interacting-object
segmentation cascade (twohands -> contact boundary -> obj1), publishing a combined
class-id mask (and optionally a colour overlay) for downstream nodes.

EgoHOS's cb/obj1 stages don't accept in-memory tensors: internally
(mmseg/models/segmentors/encoder_decoder.py) they derive the path of the previous
stage's prediction from the *file path* of the input image (sibling 'pred_twohands'
/ 'pred_cb' directories next to the image's parent dir). So each frame is written
to a scratch directory and chained through real files to match that layout.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from mmseg.apis import inference_segmentor, init_segmentor

# combined mask classes: 0 background, 1 hand (either), 2 object (in contact with either hand)
PALETTE_BGR = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
], dtype=np.uint8)

# scripts/driver.py -> EgoHOS/mmsegmentation, regardless of where the repo is checked out
EGOHOS_ROOT_DEFAULT = str(Path(__file__).resolve().parent.parent / 'mmsegmentation')


class EgoHOSNode(Node):

    def __init__(self):
        super().__init__('egohos_node')

        egohos_root = self._declare_and_get('egohos_root', EGOHOS_ROOT_DEFAULT)

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('mask_topic', 'segmentation_mask')
        self.declare_parameter('overlay_topic', 'segmentation_overlay')
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('overlay_alpha', 0.5)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('scratch_dir', '/dev/shm/h2r_handovers_egohos')

        self.declare_parameter(
            'twohands_config',
            os.path.join(egohos_root, 'work_dirs/seg_twohands_ccda/seg_twohands_ccda.py'))
        self.declare_parameter(
            'twohands_checkpoint',
            os.path.join(egohos_root, 'work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth'))
        self.declare_parameter(
            'cb_config',
            os.path.join(egohos_root, 'work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py'))
        self.declare_parameter(
            'cb_checkpoint',
            os.path.join(egohos_root, 'work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth'))
        self.declare_parameter(
            'obj1_config',
            os.path.join(egohos_root, 'work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py'))
        self.declare_parameter(
            'obj1_checkpoint',
            os.path.join(egohos_root, 'work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth'))

        device = self.get_parameter('device').value
        scratch_dir = self.get_parameter('scratch_dir').value
        self._overlay_alpha = self.get_parameter('overlay_alpha').value
        self._publish_overlay = self.get_parameter('publish_overlay').value

        self._image_dir = os.path.join(scratch_dir, 'images')
        self._twohands_dir = os.path.join(scratch_dir, 'pred_twohands')
        self._cb_dir = os.path.join(scratch_dir, 'pred_cb')
        for d in (self._image_dir, self._twohands_dir, self._cb_dir):
            os.makedirs(d, exist_ok=True)
        self._frame_path = os.path.join(self._image_dir, 'frame.jpg')
        self._twohands_path = os.path.join(self._twohands_dir, 'frame.png')
        self._cb_path = os.path.join(self._cb_dir, 'frame.png')

        self.get_logger().info(f'Loading EgoHOS models on {device}...')
        self._twohands_model = init_segmentor(
            self.get_parameter('twohands_config').value,
            self.get_parameter('twohands_checkpoint').value, device=device)
        self._cb_model = init_segmentor(
            self.get_parameter('cb_config').value,
            self.get_parameter('cb_checkpoint').value, device=device)
        self._obj1_model = init_segmentor(
            self.get_parameter('obj1_config').value,
            self.get_parameter('obj1_checkpoint').value, device=device)
        self.get_logger().info('EgoHOS models loaded.')

        self._bridge = CvBridge()
        self._busy = False

        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._on_image, camera_qos)

        self._mask_pub = self.create_publisher(
            Image, self.get_parameter('mask_topic').value, 1)
        if self._publish_overlay:
            self._overlay_pub = self.create_publisher(
                Image, self.get_parameter('overlay_topic').value, 1)

    def _declare_and_get(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _on_image(self, msg: Image):
        if self._busy:
            return
        self._busy = True
        try:
            self._process(msg)
        finally:
            self._busy = False

    def _process(self, msg: Image):
        t0 = time.monotonic()
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(self._frame_path, bgr)

        twohands_seg = inference_segmentor(self._twohands_model, self._frame_path)[0]
        cv2.imwrite(self._twohands_path, twohands_seg.astype(np.uint8))

        cb_seg = inference_segmentor(self._cb_model, self._frame_path)[0]
        cv2.imwrite(self._cb_path, cb_seg.astype(np.uint8))

        obj1_seg = inference_segmentor(self._obj1_model, self._frame_path)[0]

        mask = np.zeros_like(twohands_seg, dtype=np.uint8)
        mask[twohands_seg > 0] = 1
        mask[obj1_seg > 0] = 2

        mask_msg = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header
        self._mask_pub.publish(mask_msg)

        if self._publish_overlay:
            overlay_color = PALETTE_BGR[mask]
            alpha = self._overlay_alpha
            overlay = (bgr.astype(np.float32) * (1 - alpha)
                       + overlay_color.astype(np.float32) * alpha).astype(np.uint8)
            overlay_msg = self._bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = msg.header
            self._overlay_pub.publish(overlay_msg)

        self.get_logger().info(f'EgoHOS inference took {time.monotonic() - t0:.3f}s', throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = EgoHOSNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
