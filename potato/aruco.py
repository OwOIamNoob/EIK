import cv2
import numpy as np
from cv2 import aruco

""" This file read an image containing an ArUCO and estimate its pose
    Reversing the pose can retrieve the camera pose according to the ArUCO"""

    # realsense intrinsic
realsense_intrinsics = np.array([   [609.11572266,   0.,         318.76638794],
                                    [  0.,         608.21179199, 241.98466492],
                                    [  0.,           0.,           1.        ]], dtype=np.float32)
def aruco_detection(img, debug = False):
    output = None
    params = aruco.DetectorParameters()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    markerIds = cv2.Mat(np.array([]))
    markerCorners = cv2.Mat(np.array([]))
    rejected = cv2.Mat(np.array([]))
    detector = aruco.ArucoDetector(dictionary, params)
    ids, corners, rejected = detector.detectMarkers(img, markerCorners, markerIds, rejected)
    if debug is True:
        output = img.copy()
        aruco.drawDetectedMarkers(output, ids, corners, borderColor=(255, 0, 0))
    
    return corners, ids, output


def pose_estimate(markerCorners, intrinsics, dist, pov=None, marker_size=10):
    rvecs = []
    tvecs = []
    poses = []
    if pov is None:
            pov = np.full((4, 3), 0, np.float32)
            pov[2:, 0] = marker_size 
            pov[1:3, 1] = marker_size
            print(pov)
    for corners in markerCorners:
        print(corners[0].shape, pov.shape)
        poses.append(cv2.solvePnP(pov,  corners, intrinsics, dist)[1:])
    return poses


def pov(src, target):
    """ Transfering between src pov and target pov"""
    pass 

if __name__ == "__main__":
    dist = np.zeros(5, dtype=np.float32)


    img = cv2.imread("EIK/data/img.jpg")

    markerIds, markerCorners, output = aruco_detection(img, debug=True)
    print(markerIds, markerCorners, output.shape)
    cv2.imwrite("/work/hpc/potato/EIK/data/aruco.jpg", output)

    poses = pose_estimate(markerCorners, realsense_intrinsics, dist)
    print(poses[0][0], poses[0][1])

    pose_img = output.copy()
    cv2.drawFrameAxes(pose_img, realsense_intrinsics, dist, poses[0][0], poses[0][1], 1)
    cv2.imwrite("EIK/data/pose.jpg", pose_img)
    
    