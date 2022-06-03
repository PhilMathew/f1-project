from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from keras.utils import image_dataset_from_directory

from deepcalib import load_deepcalib_regressor, preprocess_img


# Want to ensure all the estimates are at least self-consistent (that is, they don't vary wildly).
# The video frames should all come from the same camera, so the parameters shouldn't change much.


def estimate_params_deepcalib(img_dir, weights_file):
    img_ds = image_dataset_from_directory(
        directory=str(img_dir),
        labels=None,
        batch_size=32,
        image_size=(299, 299)
    )
    preproc_ds = img_ds.map(preprocess_img)
        
    model = load_deepcalib_regressor(weights_file)
    
    focal_widths, distortion_coeffs = model.predict(preproc_ds)
    
    return focal_widths, distortion_coeffs


def graph_estimates(focal_widths, distortion_coeffs, outfile):
    fig, ((focal_ax), (dist_ax)) = plt.subplots(2, 1, figsize=(15, 10))
    
    focal_ax.plot(focal_widths)
    focal_ax.set_title('Focal Widths')
    
    dist_ax.plot(distortion_coeffs)
    dist_ax.set_title('Distortion Coefficients')
    
    fig.savefig(str(outfile))


def main():
    frames_dir = Path('./video_output')
    weights_file = Path('./weights_10_0.02.h5')
    
    focal_widths, distortion_coeffs = estimate_params_deepcalib(frames_dir, weights_file)
    graph_estimates(focal_widths, distortion_coeffs, 'consistency_test.png')


if __name__ == '__main__':
    main()
