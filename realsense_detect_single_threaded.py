import rospy
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import Image
import ros_numpy

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from utils import detector_utils as detector_utils
import tensorflow as tf
import datetime
import argparse
from scipy import ndimage
import numpy as np
from IPython import embed


detection_graph, sess = detector_utils.load_inference_graph()
pub_bbx = rospy.Publisher('hand_bbx', Float64MultiArray, queue_size=1)

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue
rgb_img = []
depth_img = []

focalLengthX = 624.3427734375
focalLengthY = 624.3428344726562

centerX = 305.03887939453125
centerY = 244.86605834960938


def callback(rgb_msg, depth_msg):
    global rgb_img, depth_img
    try:
        rgb_img = ros_numpy.numpify(rgb_msg)
        depth_img = ros_numpy.numpify(depth_msg)
    except CvBridgeError as e:
        rospy.logerr(e)


def calculateCoM(dpt):
    """
    Calculate the center of mass
    :param dpt: depth image
    :return: (x,y,z) center of mass
    """

    dc = dpt.copy()
    dc[dc < 0] = 0
    dc[dc > 10000] = 0
    cc = ndimage.measurements.center_of_mass(dc > 0)
    num = np.count_nonzero(dc)
    com = np.array((cc[1]*num, cc[0]*num, dc.sum()), np.float)

    if num == 0:
        return np.array((0, 0, 0), np.float)
    else:
        return com/num


def depth2pc(depth):
    points = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            Z = int(depth[v, u])
            if Z == 0:
                continue
            X = int((u - centerX) * Z / focalLengthX)
            Y = int((v - centerY) * Z / focalLengthY)
            points.append([X, Y, Z])
    points_np = np.array(points)
    return points_np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.5,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    rospy.init_node('hand_track_arm')

    pub_bbx = rospy.Publisher('hand_bbx', Float64MultiArray, queue_size=1)

    depth_sub = message_filters.Subscriber(
        '/camera/aligned_depth_to_color/image_raw', Image)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)

    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.sleep(1)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (640, 480)
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.random.rand(100000, 3))
    # pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(100000)])
    #
    # vis.add_geometry(pcd)

    while True:
        try:
            image_np = rgb_img
            depth_np = depth_img
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)
        ind = np.argmax(scores)
        bbx = boxes[ind]
        (left, right, top, bottom) = (bbx[1] * im_width, bbx[3] * im_width,
                                      bbx[0] * im_height, bbx[2] * im_height)
        depth_crop = depth_np[int(top):int(bottom), int(left):int(right)]

        mass_center = calculateCoM(depth_crop)
        print(np.mean(depth_crop))
        points = depth2pc(np.array(mass_center[2]).reshape(1, 1)).tolist()
        print(points)

        # embed()
        if len(points):
            msg = Float64MultiArray()
            msg.data = points[0]
            pub_bbx.publish(msg)

        # show depth image
        # n1 = cv2.normalize(depth_np, depth_np, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)
        cv2.circle(image_np, (int(mass_center[0])+int(left), int(mass_center[1])+int(top)),
         5, (0, 255, 0), -1)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
