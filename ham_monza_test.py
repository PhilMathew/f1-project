from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 


def get_sift_features(img):
    # # First convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # We use SIFT to get the feature descriptors
    sift = cv2.SIFT_create()
    
    return sift.detectAndCompute(img, None)


def find_matches(img1, img2, dist_thresh):
    pts1, descs1 = get_sift_features(img1)
    pts2, descs2 = get_sift_features(img2)
    
    bf = cv2.BFMatcher()
    match_pairs = bf.knnMatch(descs1, descs2, k=2)
    
    matches = []
    for m, n in match_pairs:
        if m.distance < dist_thresh * n.distance:
            matches.append(m)
    
    img3 = cv2.drawMatches(img1, pts1, img2, pts2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    matches1 = np.array([pts1[m.queryIdx].pt for m in matches])
    matches2 = np.array([pts2[m.trainIdx].pt for m in matches])
    
    return matches1, matches2, img3


def calc_path(frame_dir, output_dir, fmax=500):
    frame_dir, output_dir = Path(frame_dir).absolute(), Path(output_dir).absolute()
    if not output_dir.exists():
        output_dir.mkdir()
    
    sift_dir = output_dir / 'sift_test'
    if not sift_dir.exists():
        sift_dir.mkdir()
        
    # Very hacky
    f, cx, cy = 3146.913, 960, 540
    K = np.array([[f, 0, cx], 
                  [0, f, cy], 
                  [0, 0, 1]])
    
    # Get image paths and combine successive frames into frame pairs for processing
    if fmax is not None:
        imgs = sorted([str(frame_dir / f) for f in frame_dir.iterdir()][:fmax])
    else:
        imgs = sorted([str(frame_dir / f) for f in frame_dir.iterdir()])
    frame_pairs = zip(imgs[0:-1], imgs[1:])
    
    R_list, t_list = [], []
    print('Computing frame transforms...')
    for i, (img_path1, img_path2) in enumerate(tqdm(frame_pairs, total=(len(imgs) - 1))):
        # Load images
        img1, img2 = cv2.imread(img_path1), cv2.imread(img_path2)
        
        # Compute and graph matched features
        matches1, matches2, matches_img = find_matches(img1, img2, .7)
        if i % 100 == 0:
            cv2.imwrite(str(sift_dir / f'sift{i}.png'), matches_img)
        
        # Use RANSAC to compute the fundamental matrix
        F, _ = cv2.findFundamentalMat(matches1, matches2, cv2.FM_RANSAC)
        
        # Calculate essential matrix from fundamental matrix
        E = K.T @ F @ K # both frames from same camera, so K' = K
        
        # Get R, t from E
        _, R, t, _ = cv2.recoverPose(E, matches1, matches2, K)
        R_list.append(R)
        t_list.append(t)
            
    car_path = [np.array([0, 0, 0])]
    T = np.identity(4)
    for i, (R, t) in enumerate(tqdm(zip(R_list, t_list), total=len(R_list))):
        # Convert to homogeneous coordinates (tacks on row of 0s to R and an extra 1 to t)
        R = np.concatenate([R, np.zeros((1, R.shape[1]))])
        t = np.concatenate([t, np.array([[1]])])
        
        # Augment matrices and compute inverse transform
        T = np.concatenate([R, t], axis=1) @ T
        T_inv = np.linalg.inv(T)

        # Compute transformation of frame center into the first frame's coordinate space
        next_pt = T_inv @ np.array([0, 0, 0, 1])        
        car_path.append(next_pt[0:-1] / next_pt[-1])
    car_path = np.array(car_path)
    
    # Plot the car's path top-down and in 3D
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(1, 1, 1, projection='3d')
    ax_3d.plot(car_path[:, 0], car_path[:, 1], car_path[:, 2])
    
    fig_2d, ax_2d = plt.subplots(1, 1)
    ax_2d.plot(car_path[:, 0], car_path[:, 2])
    
    fig_3d.savefig(f'{output_dir}/car_path_3d.png')
    fig_2d.savefig(f'{output_dir}/car_path_2d.png')
    
    plt.show()
    
    print(f'Path plotted, see {output_dir}')
    
    
def main():
    calc_path('./video_output', './test_results', fmax=1500)


if __name__ == '__main__':
    main()
