#!/usr/bin/python
# seqproc_common.py - contains helper functions for the Sequence Processor


# imports
import math
import random
import numpy as np


# augmentation helper: find the bounding box of the region of interest
def find_window_limits(features_base):
    xmin = 1
    ymin = 1
    xmax = 0
    ymax = 0
    # do for all items in all frames
    for frameno in range(features_base.shape[0]):
        for itemno in range(features_base.shape[1]):
            # only continue if confidence is higher than 0, i.e., object is detected
            if features_base[frameno][itemno][4] > 0:
                # update region of interest if coordinate exceeds limits
                xmax = max(xmax,features_base[frameno][itemno][0]+0.5*features_base[frameno][itemno][2])
                xmin = min(xmin,features_base[frameno][itemno][0]-0.5*features_base[frameno][itemno][2])
                ymax = max(ymax,features_base[frameno][itemno][1]+0.5*features_base[frameno][itemno][3])
                ymin = min(ymin,features_base[frameno][itemno][1]-0.5*features_base[frameno][itemno][3])
    return xmin, xmax, ymin, ymax


# augmentation helper: move all boxes with a pre-defined shift
def move_boxes(features_base, shift_w, shift_h):
    # do for all items in all frames
    for frameno in range(features_base.shape[0]):
        for itemno in range(features_base.shape[1]):
            # only continue if confidence is higher than 0, i.e., object is detected
            if features_base[frameno][itemno][4] > 0:
                # shift C_X and C_Y
                features_base[frameno][itemno][0] = features_base[frameno][itemno][0] + shift_w
                features_base[frameno][itemno][1] = features_base[frameno][itemno][1] + shift_h
    return features_base


# returns a feature engineered window (in the form of a list) given the video resolution and a list of jsons
def window (res_x, res_y, json_batch, augment):

    # variables
    objects = ['cabin', 'forearm', 'upperarm', 'wheelbase', 'attachment-bucket', 'attachment-breaker']
    num_frames = len(json_batch)
    num_objects = len(objects)
    num_base_features = 5

    # construct base features from json-data
    features_base = np.zeros((num_frames, num_objects, num_base_features))
    # do for each frame
    for frameno, frame in enumerate(json_batch):
        # get the strongest detection for each category
        object_dict = {}
        for detected_object in frame['body']['predictions'][0]['classes']:
            category = detected_object['cat']
            if category in object_dict:
                if object_dict[category]['prob'] < detected_object['prob']:
                    object_dict[category] = detected_object
            else:
                object_dict[category] = detected_object
        # write feature engineered window to array
        for itemno, item in enumerate(objects):
            if item in object_dict:
                # fetch json output and translate to relative positions
                obj = object_dict[item]
                xmin = min(obj['bbox']['xmin'], obj['bbox']['xmax']) / res_x
                xmax = max(obj['bbox']['xmin'], obj['bbox']['xmax']) / res_x
                ymin = min(obj['bbox']['ymin'], obj['bbox']['ymax']) / res_y
                ymax = max(obj['bbox']['ymin'], obj['bbox']['ymax']) / res_y
                conf = obj['prob']
            else:
                xmin = 0
                xmax = 0
                ymin = 0
                ymax = 0
                conf = 0
            # define features
            C_X = (xmax - xmin)/2 + xmin
            C_Y = (ymax - ymin)/2 + ymin
            W = xmax - xmin
            H = ymax - ymin
            features = [C_X, C_Y, W, H, conf]
            # update feature array
            features_base[frameno][itemno][:] = features

    # if we want to augment, generate a random permutation now
    if augment:
        # horizontal flip (50% chance)
        if random.random() < .5:
            for frameno in range(num_frames):
                for itemno in range(num_objects):
                    if (features_base[frameno][itemno][4] != 0):
                        # only flip if object exists (conf > 0)
                        features_base[frameno][itemno][0] = 1 - features_base[frameno][itemno][0]
        # place objects in top left corner
        xmin, xmax, ymin, ymax = find_window_limits(features_base)
        move_boxes(features_base, -xmin, -ymin)
        # resize objects
        # 1.3 and 0.7 are arbitrary min/max thresholds
        xmin, xmax, ymin, ymax = find_window_limits(features_base)
        maxscale = min(1/xmax, 1/ymax, 1.3)
        minscale = 0.7
        scale = random.uniform(minscale, maxscale)
        for frameno in range(num_frames):
            for itemno in range(num_objects):
                for featureno in range(num_base_features-1):
                    features_base[frameno][itemno][featureno] = features_base[frameno][itemno][featureno] * scale
        # move objects
        xmin, xmax, ymin, ymax = find_window_limits(features_base)
        shift_w = random.uniform(0, 1-xmax)
        shift_h = random.uniform(0, 1-ymax)
        row = move_boxes(features_base, shift_w, shift_h)

    # construct difference features from base features
    features_base_diff = np.zeros((num_frames-1, num_objects, num_base_features))
    for frameno in range(num_frames-1):
        for itemno in range(num_objects):
            for featureno in range(num_base_features):
                # calculate based on previous existing object, to skip detection disappearances
                prev_existing_object = -1
                for i in range(frameno, -1, -1):
                    if features_base[i][itemno][4] != 0:
                        prev_existing_object = i
                # only calculate difference if object exists and we can find an existing predecessor (conf > 0)
                if (features_base[frameno+1][itemno][4] != 0 and prev_existing_object != -1):
                    features_base_diff[frameno][itemno][featureno] = features_base[frameno+1][itemno][featureno] - features_base[frameno][itemno][featureno]

    # for each object, calculate a motility score over whole window
    motility = np.zeros((num_objects))
    for itemno in range(num_objects):
        for frameno in range(num_frames-1):
            motility[itemno] += features_base_diff[frameno][itemno][0]
            motility[itemno] += features_base_diff[frameno][itemno][1]
        motility[itemno] /= 2*(num_frames-1)

    # relative motility of arm boxes vs cabin boxes
    relative_motility = (motility[1] + motility[2] - motility[0] - motility[3]) / 2

    # for each object in each frame, calculate the pythagorean, x and y distance to the cabin
    cabindistance = np.zeros((num_frames, num_objects-1, 3))
    for frameno in range(num_frames-1):
        for itemno in range(num_objects-1):
            x1 = features_base[frameno][0][0]
            y1 = features_base[frameno][0][1]
            x2 = features_base[frameno][itemno+1][0]
            y2 = features_base[frameno][itemno+1][1]
            cabindistance[frameno][itemno][0] = math.hypot(x2-x1,y2-y1) / 2
            cabindistance[frameno][itemno][1] = x2-x1
            cabindistance[frameno][itemno][2] = y2-y1

    # indicators of breaker / bucket window
    breaker_vs_bucket = np.zeros((4))
    for frameno in range(num_frames):
        # confidence difference
        breaker_vs_bucket[0] += features_base[frameno][4][4]
        breaker_vs_bucket[0] -= features_base[frameno][4][4]
        # forearm distance
        x_forearm = features_base[frameno][1][0]
        y_forearm = features_base[frameno][1][1]
        x_breaker = features_base[frameno][4][0]
        y_breaker = features_base[frameno][4][1]
        x_bucket = features_base[frameno][5][0]
        y_bucket = features_base[frameno][5][1]
        breaker_vs_bucket[1] += math.hypot(x_bucket-x_forearm,y_bucket-y_forearm) / 2
        breaker_vs_bucket[1] -= math.hypot(x_breaker-x_forearm,y_breaker-y_forearm) / 2
        breaker_vs_bucket[2] += x_bucket-x_forearm
        breaker_vs_bucket[2] -= x_breaker-x_forearm
        breaker_vs_bucket[3] += y_bucket-y_forearm
        breaker_vs_bucket[3] -= y_breaker-y_forearm
    # normalize output
    breaker_vs_bucket[0] /= num_frames
    breaker_vs_bucket[1] /= num_frames
    breaker_vs_bucket[2] /= num_frames
    breaker_vs_bucket[3] /= num_frames

    # total size of each object, in width, height and total size
    object_size = np.zeros((num_objects, 3))
    for itemno in range(num_objects):
        for frameno in range(num_frames):
            object_size[itemno][0] += features_base[frameno][itemno][2]
            object_size[itemno][1] += features_base[frameno][itemno][3]
            object_size[itemno][2] += features_base[frameno][itemno][2] * features_base[frameno][itemno][3]
        object_size[itemno][0] /= num_frames
        object_size[itemno][1] /= num_frames
        object_size[itemno][2] /= num_frames

    # relative size of arm boxes vs cabin boxes
    object_size_relative = np.zeros((3))
    object_size_relative[0] = (object_size[1][0] + object_size[2][0] - object_size[0][0] - object_size[3][0]) / 2
    object_size_relative[1] = (object_size[1][1] + object_size[2][1] - object_size[0][1] - object_size[3][1]) / 2
    object_size_relative[2] = (object_size[1][2] + object_size[2][2] - object_size[0][2] - object_size[3][2]) / 2

    # cumulative size warping of each object
    object_size_warping = np.zeros((num_objects, 3))
    for itemno in range(num_objects):
        for frameno in range(num_frames-1):
            object_size_warping[itemno][0] += features_base_diff[frameno][itemno][2]
            object_size_warping[itemno][1] += features_base_diff[frameno][itemno][3]
            object_size_warping[itemno][2] += features_base_diff[frameno][itemno][2] * features_base_diff[frameno][itemno][3]
        object_size_warping[itemno][0] /= num_frames
        object_size_warping[itemno][1] /= num_frames
        object_size_warping[itemno][2] /= num_frames

    # collect all engineered features
    output = \
        features_base.flatten().tolist() + \
        features_base_diff.flatten().tolist() + \
        motility.flatten().tolist() + \
        relative_motility.flatten().tolist() + \
        cabindistance.flatten().tolist() + \
        breaker_vs_bucket.flatten().tolist() + \
        object_size.flatten().tolist() + \
        object_size_relative.flatten().tolist() + \
        object_size_warping.flatten().tolist()

    # check that all values are normalized correctly
    assert(all(i >= -1 for i in output))
    assert(all(i <= 1 for i in output))

    # return window
    return output