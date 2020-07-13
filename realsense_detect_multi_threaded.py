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
import open3d as o3d

tf.debugging.set_log_device_placement(True)

frame_processed = 0
score_thresh = 0.2


# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue
rgb_img = []
depth_img = []

focalLengthX = 624.3427734375
focalLengthY = 624.3428344726562

centerX = 305.03887939453125
centerY = 244.86605834960938

cube_size = [200, 200, 200]

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


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


def clean_depth_map(depth, com, size, com_type="2D"):
    if com_type == "2D":
        com3d = [(com[0] + int(left) - depth.shape[1]/2) * com[2] / focalLengthX,
                 (com[1] + int(top) - depth.shape[0]/2) * com[2] / focalLengthY, com[2]]
    else:
        com3d = com
    x_min = com3d[0] - size[0] / 2
    x_max = com3d[0] + size[0] / 2
    y_min = com3d[1] - size[1] / 2
    y_max = com3d[1] + size[1] / 2
    z_min = com3d[2] - size[2] / 2
    z_max = com3d[2] + size[2] / 2

    points = depth2pc(depth, True, left, top)
    points_tmp = points.copy()
    if len(points):
        hand_points_ind = np.all(
        np.concatenate((points[:, 0].reshape(-1, 1) > x_min, points[:, 0].reshape(-1, 1) < x_max,
                        points[:, 1].reshape(-1, 1) > y_min, points[:, 1].reshape(-1, 1) < y_max,
                        points[:, 2].reshape(-1, 1) > z_min, points[:, 2].reshape(-1, 1) < z_max), axis=1), axis=1)
        points_tmp = points[hand_points_ind]
        depth = pc2depth(points[hand_points_ind])
    return points_tmp, depth


def jointsImgTo3D(sample):
    """
    Normalize sample to metric 3D
    :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized joints in mm
    """
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = jointImgTo3D(sample[i])
    return ret


def jointImgTo3D(sample):
    """
    Normalize sample to metric 3D
    :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized joints in mm
    """
    ret = np.zeros((3,), np.float32)
    # convert to metric using f
    ret[0] = (sample[0]-centerX)*sample[2]/focalLengthX
    ret[1] = (sample[1]-centerY)*sample[2]/focalLengthY
    ret[2] = sample[2]
    return ret


def depth2pc(depth, after_crop=False, left=0, top=0):
    points = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            Z = int(depth[v, u])
            if Z == 0:
                continue
            v_m = v
            u_m = u
            if after_crop:
                v_m = v + int(top)
                u_m = u + int(left)
            X = int((u_m - centerX) * Z / focalLengthX)
            Y = int((v_m - centerY) * Z / focalLengthY)
            points.append([X, Y, Z])
    points_np = np.array(points)
    return points_np


def pc2depth(pc_local):
    pc = pc_local.copy()
    width = 640
    height = 480
    pc[:, 0] = pc[:, 0] / pc[:, 2].astype(float) * focalLengthX + centerX
    pc[:, 1] = pc[:, 1] / pc[:, 2].astype(float) * focalLengthY + centerY
    uvd = []
    for i in range(pc.shape[0]):
        if 0 < pc[i, 0] < width and 0 < pc[i, 1] < height:
            uvd.append(pc[i, :].astype(int))
    depth = uvd2depth(np.array(uvd), width, height)
    return depth


def depth2uvd(depth):
    depth = depth.squeeze()
    v, u = np.where(depth != 0)
    v = v.reshape(-1, 1)
    u = u.reshape(-1, 1)
    return np.concatenate([u, v, depth[v, u]], axis=1)


def uvd2depth(uvd, width, height):
    depth = np.zeros((height, width, 1), np.uint16)
    depth[uvd[:, 1], uvd[:, 0]] = uvd[:, 2].reshape(-1, 1)
    return depth


def joint3DToImg(sample):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: joints in (x,y,z) with x,y and z in mm
    :return: joints in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = np.zeros((3,), np.float32)
    # convert to metric using f
    if sample[2] == 0.:
        ret[0] = centerX
        ret[1] = centerY
        return ret
    ret[0] = sample[0]/sample[2]*focalLengthX+centerX
    ret[1] = sample[1]/sample[2]*focalLengthY+centerY
    ret[2] = sample[2]
    return ret


def worker(input_q, depth_q, output_q, cap_params, frame_processed):
    global rgb_img, depth_img

    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    im_width, im_height = (640, 480)

    pcd = o3d.geometry.PointCloud()
    pcd_crop = o3d.geometry.PointCloud()
    inlier_cloud = o3d.geometry.PointCloud()

    previous_center_point = np.array([0, 0, 0])
    while True:
        # print("> ===== in worker loop, frame ", frame_processed)
        image_np = input_q.get()
        depth_np = depth_q.get()
        if image_np is not None:
            # Actual detection. Variable boxes contains the bounding box coordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found at least one hand (within your score threshold)
            #
            boxes, scores = detector_utils.detect_objects(
                image_np, detection_graph, sess)
            # boxes, scores = detector_utils.gpu_detect_objects(image_np,
            #                                               detection_graph, sess)
            ind = np.argmax(scores)
            bbx = boxes[ind]
            (left, right, top, bottom) = (bbx[1] * im_width, bbx[3] * im_width,
                                          bbx[0] * im_height, bbx[2] * im_height)
            depth_crop = depth_np[int(top):int(bottom), int(left):int(right)]
            # depth_crop[np.where(depth_crop > 900)] = 0
            points_crop = depth2pc(depth_crop, True, int(left), int(top))
            if len(points_crop):
                pcd_crop.points = o3d.utility.Vector3dVector(points_crop)
                cl, ind = pcd_crop.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=1.0)
                pcd.points = o3d.utility.Vector3dVector(points_crop)
                inlier_cloud = pcd_crop.select_down_sample(ind)
                center_point = inlier_cloud.get_center()

                # if center_point is far away with the previous center point
            else:
                # if no hand is detected, use previous center point
                center_point = previous_center_point

            previous_center_point = center_point
            print(center_point)
            points_q.put(center_point)

            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                image_np)
            center_img = joint3DToImg(center_point)
            cv2.circle(image_np, (int(center_img[0]), int(center_img[1])),
             5, (0, 255, 0), -1)

            # add image_np annotated with bounding box to queue
            output_q.put(image_np)
            frame_processed += 1
        else:
            output_q.put(image_np)
    sess.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source', type=int, default=0, help='Device index of the camera.')
    parser.add_argument('--num_hands',type=int,default=1,help='Max number of hands to detect.')
    parser.add_argument('--fps',type=int,default=20,help='Show FPS on detection/display visualization')
    parser.add_argument('--width',type=int,default=640,help='Width of the frames in the video stream.')
    parser.add_argument('--height',type=int,default=480,help='Height of the frames in the video stream.')
    parser.add_argument('--display',type=int,default=0,help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('--num-workers',type=int,default=1,help='Number of workers.')
    parser.add_argument('--queue-size',type=int,default=5,help='Size of the queue.')
    args = parser.parse_args()

    rospy.init_node('hand_track_arm')
    pub_bbx = rospy.Publisher('hand_bbx', Float64MultiArray, queue_size=1)
    depth_sub = message_filters.Subscriber(
        '/camera/aligned_depth_to_color/image_raw', Image)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.sleep(1)

    input_q = Queue(maxsize=args.queue_size)
    depth_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    points_q = Queue(maxsize=args.queue_size)

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
    if args.display > 0:
        cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        index += 1
        # embed()
        input_q.put(rgb_img)
        depth_q.put(depth_img)

        # worker(input_q, depth_q, output_q, cap_params, frame_processed)
        output_frame = output_q.get()
        center_points = points_q.get()
        if center_points is not None:
            msg = Float64MultiArray()
            msg.data = center_points
            print(msg)
            pub_bbx.publish(msg)
        # output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
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
    # cv2.destroyAllWindows()
