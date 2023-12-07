import open3d as o3d 
import numpy as np
import aruco 

reader = o3d.t.io.RSBagReader()
reader.open("/work/hpc/potato/EIK/potato/data/realsense.bag")
instrinsic = aruco.realsense_intrinsics
camera_spec = o3d.camera.PinholeCameraIntrinsic(640, 480, instrinsic[0, 0], instrinsic[1, 1], instrinsic[0, 2], instrinsic[1, 2])
im_rgbd = reader.next_frame()
print(im_rgbd)

color_img = np.asarray(im_rgbd.color)
depth_img = np.asarray(im_rgbd.depth)
"""" From pybind to another pybind ???? """
print(type(im_rgbd))
print("Img shape: ", color_img.shape, depth_img.shape)
markerIds, markerCorners, output = aruco.aruco_detection(color_img)
# depth scale = 1000 (1m = 1000 unit)
poses = aruco.pose_estimate(markerCorners, 
                            aruco.realsense_intrinsics, 
                            None,
                            marker_size=100)
# print(markerCorners[0][0][0])
world_depth = depth_img[int(markerCorners[0][0][0][0]), int(markerCorners[0][0][0][1])][0]
finetuned_tvec, rodvec = aruco.finetune_tvec(np.array(markerCorners[0][0][0], dtype=int),
                                           world_depth, 
                                           poses[0], 
                                           aruco.realsense_intrinsics,
                                           export_rvec=True)
print("Translation: ", finetuned_tvec, np.linalg.norm(finetuned_tvec), world_depth)
print("Rotation: ", rodvec, rodvec.shape)
homo = np.concatenate([rodvec, finetuned_tvec[np.newaxis, :].T], axis=1)
rigid = np.concatenate([homo, [[0, 0, 0, 1]]], axis = 0)
new_pose = o3d.core.Tensor(rigid)
intrinsics = o3d.core.Tensor(instrinsic)
point_cloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, 
                                                               intrinsics=intrinsics, 
                                                               extrinsics=new_pose)
print(o3d.t.io.write_point_cloud("/work/hpc/potato/EIK/potato/data/point.xyz", point_cloud))
reader.close()

