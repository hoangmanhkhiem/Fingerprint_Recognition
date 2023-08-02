import argparse
import os
import sys
import numpy as np
import json
import math

from finegerprint_pipline import *


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--finger_image', required=True,
                help='path to input image')
parser.add_argument('-t', '--threshold', required=True,)
args = parser.parse_args()

def score_matching(finger, finger_d, threshold_d, threshold_o):
    x_core = 0
    y_core = 0
    x_core_d = 0
    y_core_d = 0
    if (len(finger_d['loop']) + len(finger_d['whorl']) != 0) and (len(finger['loop']) + len(finger['whorl']) != 0):
        if (len(finger_d['loop'])) != 0:
            x_core_d = finger_d['loop'][0]
            y_core_d = finger_d['loop'][1]
        else:
            x_core_d = finger_d['whorl'][0]
            y_core_d = finger_d['whorl'][1]

        if (len(finger['loop'])) != 0:
            x_core = finger['loop'][0]
            y_core = finger['loop'][1]
        else:
            x_core = finger['whorl'][0]
            y_core = finger['whorl'][1]


    n_d = len(finger_d['loop'])/3 + len(finger_d['whorl'])/3 + len(finger_d['delta']) + len(finger_d['ending']) + len(finger_d['bifurcation'])
    n = 0

    if len(finger['loop']) != 0 and len(finger_d['loop'])!=0:
        sd_loop = math.sqrt(((finger['loop'][0]-x_core) - (finger_d['loop'][0]-x_core_d))**2 + ((finger['loop'][1]-y_core) - (finger_d['loop'][1]-y_core_d))**2)
        dd_loop = abs(finger['loop'][2] - finger_d['loop'][2])
        if sd_loop < threshold_d and dd_loop < threshold_o:
            n += 5

    if len(finger['whorl']) != 0 and len(finger_d['whorl'])!=0:
        sd_whorl = math.sqrt(((finger['whorl'][0]-x_core) - (finger_d['whorl'][0]-x_core_d))**2 + ((finger['whorl'][1]-y_core) - (finger_d['whorl'][1])-y_core_d)**2)
        dd_whorl = abs(finger['whorl'][2] - finger_d['whorl'][2])
        if sd_whorl < threshold_d and dd_whorl < threshold_o:
            n += 5

    if len(finger['delta']) != 0 and len(finger_d['delta'])!=0:
        for delta in finger['delta']:
            for delta_d in finger_d['delta']:
                sd_delta = math.sqrt(((delta[0]-x_core) - (delta_d[0]-x_core_d))**2 + ((delta[1]-y_core) - (delta_d[1]-y_core_d))**2)
                dd_delta = abs(delta[2] - delta_d[2])
                if sd_delta < threshold_d and dd_delta < threshold_o:
                    n=n+5

    ending_d = finger_d['ending']
    for end_d in ending_d:
        for end in finger['ending']:
            sd_end = math.sqrt(((end[0]-x_core) - (end_d[0]-x_core_d))**2 + ((end[1]-y_core) - (end_d[1]-y_core_d))**2)
            dd_end = abs(end[2] - end_d[2])
            if sd_end < threshold_d and dd_end < threshold_o:
                n=n+1
                end_d = [0,0,0]##can phai sua lai

    bifurcation_d = finger_d['bifurcation']
    for bif_d in bifurcation_d:
        for bif in finger['bifurcation']:
            sd_bif = math.sqrt(((bif[0]-x_core) - (bif_d[0]-x_core_d))**2 + ((bif[1]-y_core) - (bif_d[1]-y_core_d))**2)
            dd_bif = abs(bif[2] - bif_d[2])
            if sd_bif < threshold_d and dd_bif < threshold_o:
                n=n+1
                bif_d = [0,0,0]## can phai sua lai

    return n/n_d
    
def matching_finger(finger, database, threshold_d, threshold_o):
    return [score_matching(finger,finger_d, threshold_d, threshold_o) for finger_d in database]
    

if __name__ == '__main__':
    threshold_d = 50
    threshold_o = 10*np.pi/180
    image = np.array(cv.imread(args.finger_image,0))
    
    results, feature_finger = fingerprint_pipline(image)
    
    name = list_name([args.finger_image])
    feature = convert_dict([feature_finger],name)[0]
    
    with open("/content/fingerprint_recognition/database/database.json", 'r') as f:
        database = json.load(f)
    
    score = matching_finger(feature, database, threshold_d, threshold_o)
    
    index_max = score.index(max(score))
    print(database[index_max]["id_name"],score[index_max])
    
    