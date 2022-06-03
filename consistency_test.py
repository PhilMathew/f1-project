from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from deepcalib import load_deepcalib_regressor, preprocess_img


# Want to ensure all the estimates are at least self-consistent (that is, they don't vary wildly).
# The video frames should all come from the same camera, so the parameters shouldn't change much.


def estimate_params_deepcalib(img_paths, weights_file):
    model = load_deepcalib_regressor(weights_file)
    
    focal_widths, distortion_coeffs = [], []
    for p in tqdm(img_paths):
        img = preprocess_img(str(p))
        focal_pred, dist_pred = model.predict(img)
        focal_widths.append(focal_pred)
        distortion_coeffs.append(dist_pred)
    
    return focal_widths, distortion_coeffs


def graph_estimates(focal_widths, distortion_coeffs, outfile):
    fig, ((focal_ax), (dist_ax)) = plt.subplots(2, 1, figsize=(15, 10))
    
    focal_ax.plot(focal_widths)
    focal_ax.set_title('Focal Widths')
    
    dist_ax.plot(distortion_coeffs)
    dist_ax.set_title('Distortion Coefficients')
    
    fig.savefig(str(outfile))


def main():
    video_frames = Path('./video_output')
    weights_file = Path('./weights_10_0.02.h5')
    img_paths = [p for p in video_frames.iterdir() if p.suffix == '.png']
    
    focal_widths, distortion_coeffs = estimate_params_deepcalib(img_paths, weights_file)
    graph_estimates(focal_widths, distortion_coeffs, 'consistency_test.png')


if __name__ == '__main__':
    main()
