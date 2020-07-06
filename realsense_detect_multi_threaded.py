from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
from multiprocessing import Queue, Pool
import datetime
import argparse
import rospy
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import Image


frame_processed = 0
score_thresh = 0.2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

pub_bbx = rospy.Publisher('hand_bbx', Float64MultiArray, queue_size=1)

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue
rgb_img = []
depth_img = []

focalLengthX = 475.065948
focalLengthY = 475.065857

centerX = 315.944855
centerY = 245.287079


def callback(rgb_msg, depth_msg):
    global rgb_img, depth_img
    try:
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    except CvBridgeError as e:
        rospy.logerr(e)


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
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found at least one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # calculate the mass center of the bbx
            mass_center = np.mean(boxes)

            # calculate the xyz of the mass center with respect to camera frame
            Z = depth[mass_center] - 0.3  # go back 30cm
            X = (u - centerX) * Z / focalLengthX
            Y = (v - centerY) * Z / focalLengthY
            msg = Float64MultiArray()
            msg.data = np.array([X, Y, Z])
            pub_bbx.pubslish(boxes)

            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            # add frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == '__main__':
    global rgb_img, depth_img
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

    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Int32)
    rgb_sub = message_filters.Subscriber('/camera/image_raw', Float32)

    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.spin()

    while True:
        # ret, oriImg = cap.read()
        # frame = oriImg
        # frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        # todo: check if need to convert to rgb
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
    # video_capture.stop()
    cap.release()
    cv2.destroyAllWindows()
