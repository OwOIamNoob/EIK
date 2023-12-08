import open3d as o3d
import numpy as np
import cv2

reader = o3d.t.io.RSBagReader()
print(reader.open("/home/potato/EIK/potato/data/realsense.bag"))
output_vid = cv2.VideoWriter('/home/potato/EIK/potato/data/depth_recorded.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         30.0, 
                         (640, 480))
while reader.is_eof() is False:
    im_rgbd = reader.next_frame()
    rgb_im = np.asarray(im_rgbd.color).astype(np.uint8)
    depth_im = np.asarray(im_rgbd.depth.filter_gaussian(kernel_size=5)) / 5
    depth_im = np.repeat(depth_im, repeats=3, axis=2).astype(np.uint8)
    try:
        frame = np.hstack((rgb_im, depth_im)).astype(np.uint8)
        print(frame.shape, frame.dtype)
        output_vid.write(frame)
    except:
        continue
output_vid.release()
