import cv2 as cv
from glob import glob
import os
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize
import json

def fingerprint_pipline(input_img):
    block_size = 16

    # pipe line picture re https://www.cse.iitk.ac.in/users/biometrics/pages/111.JPG
    # normalization -> orientation -> frequency -> mask -> filtering

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # color threshold
    # threshold_img = normalized_img
    # _, threshold_im = cv.threshold(normalized_img,127,255,cv.THRESH_OTSU)
    # cv.imshow('color_threshold', normalized_img); cv.waitKeyEx()

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)

    # minutias
    minutias, info_minutiaes = calculate_minutiaes(thin_image, angles, block_size)

    # singularities
    singularities_img, info_singularity = calculate_singularities(thin_image, angles, 1, block_size, mask)

    # visualize pipeline stage by stage
    output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)

    feature_finger = {
        "freq": freq,
        "info_singularity": info_singularity,
        "info_minutias": info_minutiaes,
    }

    return results, feature_finger

def convert_dict(feature_fingers,ls_name):
    output = []
    for feature_finger, name in zip(feature_fingers,ls_name):
        user = {
            'id_name': name,
            'freq': feature_finger['freq'].tolist(),
            'loop': feature_finger['info_singularity']['loop'].tolist(),
            'whorl': feature_finger['info_singularity']['whorl'].tolist(), 
            'delta': feature_finger['info_singularity']['delta'],
            'ending': feature_finger['info_minutias']['ending'],
            'bifurcation': feature_finger['info_minutias']['bifurcation'],
        }
        output.append(user)
    return output


   
def list_name(list_paths):
    return [path.split('/')[-1].split('_')[0] for path in list_paths]  

if __name__ == '__main__':
    # open images
    img_dir = '/content/fingerprint_recognition/finger_signed/*'
    output_dir = '/content/fingerprint_recognition/output/'
    def open_images(directory):
        images_paths = glob(directory)
        return np.array([cv.imread(img_path,0) for img_path in images_paths]), images_paths

    images, list_path = open_images(img_dir)

    feature_fingers = []
    # image pipeline
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(tqdm(images)):
        results, feature_finger = fingerprint_pipline(img)
        feature_fingers.append(feature_finger)
        cv.imwrite(output_dir+str(i)+'.png', results)
        # cv.imshow('image pipeline', results); cv.waitKeyEx()
    
    data_base = json.dumps(convert_dict(feature_fingers,list_name(list_path)), indent=3)
    
    with open("/content/fingerprint_recognition/database/database.json", "w") as outfile:
        outfile.write(data_base)
