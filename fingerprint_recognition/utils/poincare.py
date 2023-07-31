from utils import orientation
import math
import cv2 as cv
import numpy as np

def poincare_index_at(i, j, angles, tolerance):
    """
    compute the summation difference between the adjacent orientations such that the orientations is less then 90 degrees
    https://books.google.pl/books?id=1Wpx25D8qOwC&lpg=PA120&ots=9wRY0Rosb7&dq=poincare%20index%20fingerprint&hl=pl&pg=PA120#v=onepage&q=poincare%20index%20fingerprint&f=false
    :param i:
    :param j:
    :param angles:
    :param tolerance:
    :return:
    """
    cells = [(-1, -1), (-1, 0), (-1, 1),         # p1 p2 p3
            (0, 1),  (1, 1),  (1, 0),            # p8    p4
            (1, -1), (0, -1), (-1, -1)]          # p7 p6 p5

    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    for k in range(0, 8):

        # calculate the difference
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180

        index += difference

    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"


def calculate_singularities(im, angles, tolerance, W, mask):
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

    # DELTA: RED, LOOP:ORAGNE, whorl:INK
    colors = {"loop" : (0, 0, 255), "delta" : (0, 128, 255), "whorl": (255, 153, 255)}
    
    list_singularity = {
        "loop" : [],
        "delta": [],
        "whorl": [],
    }
    
    for i in range(3, len(angles) - 2):             # Y
        for j in range(3, len(angles[i]) - 2):      # x
            # mask any singularity outside of the mask
            mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
            mask_flag = np.sum(mask_slice)
            if mask_flag == (W*5)**2:
                singularity = poincare_index_at(i, j, angles, tolerance)
                if singularity != "none":
                    cv.rectangle(result, ((j+0)*W, (i+0)*W), ((j+1)*W, (i+1)*W), colors[singularity], 3)
                    list_singularity[singularity].append([int((j+0.5)*W),int((i+0.5)*W),angles[i,j]])

    #chuan hoa core va delta (delta phai va trai)
    list_singularity["loop"]=np.array(list_singularity["loop"])
    if len(list_singularity["loop"])!=0:
      list_singularity["loop"] = list_singularity["loop"].mean(axis=0)
    list_singularity["whorl"]=np.array(list_singularity["whorl"])
    if len(list_singularity["whorl"])!=0:
      list_singularity["whorl"] = list_singularity["whorl"].mean(axis=0)
    x_core = None
    if len(list_singularity["loop"])!=0:
        x_core = list_singularity["loop"][0]
    if len(list_singularity["whorl"])!=0:
        x_core = list_singularity["whorl"][0]
    #----------
    
    list_singularity["delta"] = np.array(list_singularity["delta"])
    if len(list_singularity["delta"]) != 0:
        if x_core != None:
            delta_left = []
            delta_right = []
            for delta in list_singularity["delta"]:
                if delta[0] < x_core:
                    delta_left.append(delta)
                if delta[0] > x_core:
                    delta_right.append(delta)
            list_delta = []
            if len(delta_left)!=0:
                delta_left = np.array(delta_left).mean(axis=0)
                list_delta.append(delta_left.tolist())
            if len(delta_left)!=0:
                delta_right = np.array(delta_right).mean(axis=0)
                list_delta.append(delta_left.tolist())
            list_singularity["delta"] = list_delta
        else:
            list_singularity["delta"]=np.array([list_singularity["delta"].mean(axis=0)]).tolist()
    return result, list_singularity
    # return result


if __name__ == '__main__':
    img = cv.imread('../test_img.png', 0)
    cv.imshow('original', img)
    angles = orientation.calculate_angles(img, 16, smoth=True)
    result, info = calculate_singularities(img, angles, 1, 16)
