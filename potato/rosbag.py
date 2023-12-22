import open3d as o3d 
import numpy as np
import aruco
import cv2 
import matplotlib.pyplot as plt
#/home/potato/EIK/potato/data/realsense.bag

# local feature engine
# Hello ?
surf =   cv2.SIFT_create(400)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) 
flann_matcher = cv2.FlannBasedMatcher(index_params,search_params)

reader = o3d.t.io.RSBagReader()
print(reader.open("/home/potato/EIK/potato/data/realsense.bag"))
instrinsic = aruco.realsense_intrinsics
camera_spec = o3d.camera.PinholeCameraIntrinsic(640, 480, instrinsic[0, 0], instrinsic[1, 1], instrinsic[0, 2], instrinsic[1, 2])
i = 0
world = None
total_poses = []
prev_des = None
keys_prev = None
depth_prev = None
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 10))
# output_vid = cv2.VideoWriter('/home/potato/EIK/potato/data/recorded.avi', 
#                          cv2.VideoWriter_fourcc(*'XVID'), 
#                          30.0, 
#                          (640, 480))
# print(type(output_vid))
while reader.is_eof() is False:
    print("Processing {}-th frame".format(i))
    i += 1
    im_rgbd = reader.next_frame()
    color_img = np.asarray(im_rgbd.color)[:, :, ::-1].copy()
    # try:
    #     output_vid.write(color_img.astype(np.uint8))
    # except:
    #     print("Sthg wrong")
    #     continue
    if i % 5 != 0:
        continue
    if i >= 1100:
        break
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
    # stitching and things, more like constraining the update
    kps, des = surf.detectAndCompute(color_img, None)
    matches = []
    if prev_des is None:
        prev_des = des
        keys_prev = np.float32([kp.pt for kp in kps])
        depth_prev = depth_img
    else:
        raw = flann_matcher.knnMatch(prev_des, des, k=3)
        for src in raw:
            if src[0].distance < 0.7 * src[1].distance:
                matches.append(src[0]) 
        keys = np.float32([kp.pt for kp in kps])
        pts1 = np.float32([keys[m.trainIdx] for m in matches])
        pts_prev = np.float32([keys_prev[m.queryIdx] for m in matches])
        mtx = cv2.findHomography(pts_prev, pts1)[0]
        print(mtx)
        warped_mask = cv2.warpPerspective(depth_prev, 
                                          mtx, 
                                          (640, 480),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=10)
        warped_mask = np.abs(warped_mask - depth_img.squeeze())
        warped_mask = (warped_mask < 0.02).astype(np.float32)
        # plt.imshow((warped_mask > 0.5).astype(np.uint8))
        # plt.show()
        if np.average(warped_mask) < 0.3:
            print("Change pov")
            depth_prev = depth_img.copy()
            keys_prev = keys.copy()
            prev_des = des.copy()
        depth_img[warped_mask > 0.5] = 200
        
        # cv2.drawContours(depth_img, convex, -1, color=(20, ), thickness=cv2.FILLED)

    depth_img = o3d.core.Tensor((depth_img * 1000).astype(np.uint16))
    im_rgbd.depth = o3d.t.geometry.Image(depth_img)
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
                                                                depth_max=10.)
    if world is None:
        world = point_cloud.clone()
    else:
        print("Adding")
        world = world.append(point_cloud)
        world.voxel_down_sample(voxel_size=0.05)
# output_vid.release()
pos = np.array(total_poses)
print(pos.shape)
camera_pov = np.ones((pos.shape[0], 1, 3)) @ pos[:, :, :3]
vertex = pos[:, :, 3].copy()

# world.voxel_down_sample(voxel_size=0.02)
fig = plt.subplot(111, projection='3d')
fig.plot3D(vertex[:, 0], vertex[:, 1], vertex[:, 2])
fig.scatter([0], [0], [0], linewidths=3)
o3d.visualization.draw_geometries([world.to_legacy()],
                                  zoom=0.3412,
                                  front=[0, -0.2, 0.8795],
                                  lookat=[0, 0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])
plt.show()
print(o3d.t.io.write_point_cloud("/home/potato/EIK/potato/data/point.xyzrgb", point_cloud, write_ascii=True, print_progress=True))
reader.close()

