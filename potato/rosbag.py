import open3d as o3d 
import numpy as np
import aruco
import cv2 
import matplotlib.pyplot as plt
#/home/potato/EIK/potato/data/realsense.bag
reader = o3d.t.io.RSBagReader()
print(reader.open("/home/potato/EIK/potato/data/realsense.bag"))
instrinsic = aruco.realsense_intrinsics
camera_spec = o3d.camera.PinholeCameraIntrinsic(640, 480, instrinsic[0, 0], instrinsic[1, 1], instrinsic[0, 2], instrinsic[1, 2])
i = 0
world = None
total_poses = []
output_vid = cv2.VideoWriter('/home/potato/EIK/potato/data/recorded.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, 
                         (640, 480), 
                         False)
print(type(output_vid))
while reader.is_eof() is False:
    print("Processing {}-th frame".format(i))
    i += 1
    im_rgbd = reader.next_frame()
    if i % 3 != 0:
        continue
    if i >= 100:
        break
    color_img = np.asarray(im_rgbd.color)
    output_vid.write(color_img[:, :, ::-1].astype(np.uint8))

    # cv2.imwrite("/home/potato/EIK/potato/data/realsense.jpg", color_img.astype(np.uint8))
    depth_img = np.asarray(im_rgbd.depth) / 1000
    # cv2.imwrite("/home/potato/EIK/potato/data/realsense_depth.jpg", (depth_img * 200).astype(np.uint8))
    """" From pybind to another pybind ???? """
    print(type(im_rgbd))
    print("Img shape: ", color_img.shape, depth_img.shape)
    # b, n, (x, y)
    markerIds, markerCorners, output = aruco.aruco_detection(color_img)
    if not markerIds:
        continue
    if len(markerIds) == 0:
        continue
    # depth scale = 1000 (1m = 1000 unit)
    poses = aruco.pose_estimate(markerCorners, 
                                aruco.realsense_intrinsics, 
                                None,
                                marker_size=0.1)
    # print(markerCorners[0][0][0])
    world_depth = depth_img[int(markerCorners[0][0][0][1]), int(markerCorners[0][0][0][0])][0]
    finetuned_tvec, rodvec = aruco.finetune_tvec(np.array(markerCorners[0][0][0], dtype=int),
                                            world_depth, 
                                            poses[0], 
                                            aruco.realsense_intrinsics,
                                            export_rvec=True)
    print("Translation: ", "Old:", poses[0][1], "New", finetuned_tvec, np.linalg.norm(finetuned_tvec), world_depth)
    print("Rotation: ", rodvec, rodvec.shape)
    homo = np.concatenate([rodvec, -finetuned_tvec[np.newaxis, :].T], axis=1)
    total_poses.append(homo)
    rigid = np.concatenate([homo, [[0, 0, 0, 1]]], axis = 0)
    new_pose = o3d.core.Tensor(rigid)
    intrinsics = o3d.core.Tensor(instrinsic)
    point_cloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, 
                                                                intrinsics=intrinsics, 
                                                                extrinsics=new_pose,
                                                                depth_max=20.)
    if world is None:
        world = point_cloud.clone()
    else:
        print("Adding")
        world = world.append(point_cloud)
        world.voxel_down_sample(voxel_size=0.05)
output_vid.release()
pos = np.array(total_poses)
print(pos.shape)
camera_pov = np.ones((pos.shape[0], 1, 3)) @ pos[:, :, :3]
vertex = pos[:, :, 3].copy()

# world.voxel_down_sample(voxel_size=0.02)
fig = plt.subplot(111, projection='3d')
fig.plot3D(vertex[:, 0], vertex[:, 1], vertex[:, 2])

o3d.visualization.draw_geometries([world.to_legacy()],
                                  zoom=0.3412,
                                  front=[0.4257, 0, 0.8795],
                                  lookat=[0, 0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])
plt.show()
print(o3d.t.io.write_point_cloud("/home/potato/EIK/potato/data/point.xyzrgb", point_cloud, write_ascii=True, print_progress=True))
reader.close()

