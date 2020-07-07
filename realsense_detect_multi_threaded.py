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
from multiprocessing import Queue, Pool
import datetime
import argparse

from scipy import ndimage
import numpy as np
from IPython import embed


frame_processed = 0
score_thresh = 0.2

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


def worker(input_q, depth_q, output_q, cap_params, frame_processed):
    global rgb_img, depth_img

    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        # print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        depth = depth_q.get()
        if frame is not None:
            # Actual detection. Variable boxes contains the bounding box coordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found at least one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            ind = np.argmax(scores)
            bbx = boxes[ind]
            im_width = frame.shape[1]
            im_height = frame.shape[0]
            (left, right, top, bottom) = (bbx[1] * im_width, bbx[3] * im_width,
                                          bbx[0] * im_height, bbx[2] * im_height)
            depth_crop = depth[int(top):int(bottom), int(left):int(right)]

            mass_center = calculateCoM(depth_crop)
            print(np.mean(depth_crop))
            points = depth2pc(np.array(mass_center[2]).reshape(1, 1)).tolist()
            print(points)

            # embed()
            if len(points):
                msg = Float64MultiArray()
                msg.data = points[0]
                pub_bbx.publish(msg)

            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            cv2.circle(frame, (int(mass_center[0]) + int(left), int(mass_center[1]) + int(top)),
                       5, (0, 255, 0), -1)

            # add frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=20,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=640,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=480,
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
        default=1,
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
    depth_sub = message_filters.Subscriber(
        '/camera/aligned_depth_to_color/image_raw', Image)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)

    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    # rospy.spin()
    rospy.sleep(1)

    input_q = Queue(maxsize=args.queue_size)
    depth_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = 640, 480
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, depth_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        index += 1
        # embed()
        input_q.put(rgb_img)
        depth_q.put(depth_img)

        # worker(input_q, depth_q, output_q, cap_params, frame_processed)
        output_frame = output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        # print("frame ",  index, num_frames, elapsed_time, fps)

        if output_frame is not None:
            if args.display > 0:
                if args.fps > 0:
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
                cv2.imshow('Multi-Threaded Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if num_frames == 400:
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))
        else:
            # print("video end")
            break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    cv2.destroyAllWindows()
