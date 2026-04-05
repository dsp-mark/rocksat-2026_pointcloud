from io import StringIO
import os

# Used for rectification and stuff
import cv2
# Used for more load effective lin alg
import numpy as np
# Used for displaying point cloud, like with the LiDAR!
import open3d as o3d

FOLDER_PATH = "Stereovision Camera Tests"
LEFT_PATH = os.path.join(FOLDER_PATH, "Cam_L.avi")
RIGHT_PATH = os.path.join(FOLDER_PATH, "Cam_R.avi")

# The frame count is really weird with avi for some reason, so I wanted to double-check that 
# the files actually existed.
print("LEFT exists:", os.path.exists(LEFT_PATH))
print("RIGHT exists:", os.path.exists(RIGHT_PATH))


# This is the cludge; I was having trouble getting the code to recognize the .txt file
# It's kind of clunky and basically just removes the brackets
def load_matrix_eval(filename):
    with open(filename, "r") as f:
        text = f.read()
    cleaned = text.replace("[", "").replace("]", "")
    arr = np.loadtxt(StringIO(cleaned), dtype=np.float64)
    return arr



cap_left = cv2.VideoCapture(LEFT_PATH)
cap_right = cv2.VideoCapture(RIGHT_PATH)

if not cap_left.isOpened() or not cap_right.isOpened():
    raise ValueError("Could not open one or both video files.")


frame_width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = min(int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT)))

# I was having size problems later on with the different vectors, so I just
# wanted to check the frame count, height, etc
# ultimately, I kept getting frame count of 0 but it was processing frames when I just while True
# so it seems okay, oddly.
n_left = cap_left.get(cv2.CAP_PROP_FRAME_COUNT)
n_right = cap_right.get(cv2.CAP_PROP_FRAME_COUNT)
print("Left frame count:", n_left)
print("Right frame count:", n_right)
print("total_frames:", total_frames)
print(f"Processing {total_frames} synchronized frame pairs at {frame_width}x{frame_height}")


# HERE'S THE DISTORTION MATRIX STUFF
K = load_matrix_eval("Camera Calibration/distortion_matrix_mtx.txt")
dist_raw = load_matrix_eval("Camera Calibration/distortion_matrix_dist.txt")
# Ultimate cludge; it was having trouble with reading in the dist.txt, so I reshaped, but
# it might have messed up what the dist represents; I'm unsure...
dist = dist_raw.ravel().reshape(-1, 1) if dist_raw.ndim <= 2 else dist_raw.reshape(-1, 1)

# Since we did one distortion matrix to be used by both cameras, I just used 
# the same for both
K_left = K.copy()
dist_left = dist.copy()
K_right = K.copy()
dist_right = dist.copy()

# Distance between the cameras is 5.5inches and 1 inch = 25.4 mm
baseline_mm = 5.5 * 25.4
baseline_m = baseline_mm / 1000.0

R = np.eye(3, dtype=np.float64)
T = np.array([-baseline_m, 0.0, 0.0], dtype=np.float64)


# I was having trouble with vector sizes for a bit, so I was just having it display the sizes
# to debug. Leaving it here in case it's helpful for you, too.
print("K_left shape:", K_left.shape)
print("K_left:\n", K_left)
print("dist_left shape:", dist_left.shape)

print("K_right shape:", K_right.shape)
print("dist_right shape:", dist_right.shape)


left_size = (frame_width, frame_height)


# IT'S RECTIFICATION TIME!!!
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_left, dist_left,
    K_right, dist_right,
    left_size,
    R, T,
)

map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    K_left, dist_left,
    R1, P1,
    left_size,
    cv2.CV_32FC1
)

map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    K_right, dist_right,
    R2, P2,
    left_size,
    cv2.CV_32FC1
)


matcher = cv2.StereoSGBM_create(
    # The number of disparities you're looking for
    # this is the maximum number (which might be another
    # reason why it's taking forever); it needs to be
    # divisible by 16
    numDisparities=128,
    # The size of the matching block; has to be odd
    # I was reading that 7-11 is common for realtime
    blockSize=11,

    P1=8*9*9,
    P2=32*9*9,

    # This is the maximum difference allowed between left
    # and right; this is also really high, so it might be 
    # increasing precision at the cost of recall (it's too picky maybe)
    disp12MaxDiff=1,

    # The percentage difference between the thing most common the
    # the left and right camera (when it's figuring out what point
    # in left is the same in the right); people were saying 5 to 15 is 
    # normal so I just kind of went in the middle
    uniquenessRatio=10,

    # Noise filtering; people typically use 50 to 200
    # Once again, went with middle
    speckleWindowSize=100,
) 


pcd_points = []
pcd_colors = []
frame_idx = 0

# This is the loop that takes FOREVER and breaks and needs fixing
# It's using line 135 to end the loop.
while True:
    ret_left, left_frame = cap_left.read()
    ret_right, right_frame = cap_right.read()

    if not (ret_left and ret_right):
        print(f"Stopped at frame {frame_idx} (ret_left={ret_left}, ret_right={ret_right})")
        break

    left_frame = cv2.resize(left_frame, (frame_width, frame_height))
    right_frame = cv2.resize(right_frame, (frame_width, frame_height))

    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    left_rect = cv2.remap(left_gray, map_left_x, map_left_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_gray, map_right_x, map_right_y, cv2.INTER_LINEAR)

    disparity = matcher.compute(left_rect, right_rect).astype(np.float32) / 16.0
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    valid_mask = (disparity > disparity.max() * 0.05) & (np.abs(points_3d[..., 2]) < 1e4)
    pts = points_3d[valid_mask].reshape(-1, 3)
    cols = colors[valid_mask].reshape(-1, 3)

    pcd_points.append(pts)
    pcd_colors.append(cols)


    # I wanted to make sure the minimum, maximum, and etc were actually calculating 
    # reasonable values
    print(f"Disparity MIN: {disparity.min():.4f} MAX: {disparity.max():.4f}")
    print(f"Disparity COUNT: {np.count_nonzero(np.isfinite(disparity))}")
    print(f"Valid Masks: {np.count_nonzero(valid_mask)}")

    # This always results in division by zero since total_frames is always zero for some reason
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx}/{total_frames}")


# Finished going through the video and calculating depth!
cap_left.release()
cap_right.release()


## ONTO VISIUALIZATION
## This part is just creating the pointcloud that's going to be displayed
all_points = np.vstack(pcd_points) if pcd_points else np.empty((0, 3))
all_colors = np.vstack(pcd_colors) if pcd_colors else np.empty((0, 3))


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.colors = o3d.utility.Vector3dVector(all_colors)

# This is what actually displays; it's supposed to create a window and show the pointcloud in 3D
# It's the same thing the LiDAR uses.
pcd = pcd.voxel_down_sample(voxel_size=0.005)
o3d.io.write_point_cloud("stereo_pointcloud.ply", pcd)
o3d.visualization.draw_geometries([pcd], window_name="Stereovision Point Cloud")