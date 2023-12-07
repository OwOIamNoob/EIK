import cv2
import numpy as np
from cv2 import aruco
import open3d as o3d
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

def get_fov(res, mtx):
    return np.atan(res[0] / (2 * mtx[1][1]), res[1] / (2 * mtx[0][0]))

def finetune_tvec(world_center, depth, pose, intrinsics, export_rvec=False):
    rpos, tvec = pose
    rvec = cv2.Rodrigues(rpos)[0]
    print(rvec.shape)
    print(world_center)
    dvec = (world_center - intrinsics[2, :2]) / [intrinsics[1, 1], intrinsics[0, 0]]
    if dvec.shape[0] < 3:
        dvec = np.pad(dvec, (0, 1), 'constant', constant_values=(1,))
    print(dvec, rvec)
    dvec = rvec @ dvec
    dvec = dvec / np.linalg.norm(dvec)
    return dvec * depth, rvec if export_rvec is True else None

def deproject(rgbd_img, intrinsic, extrinsic):
    return o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_img, 
                                                           intrinsic, 
                                                           extrinsic)
    


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
    
    