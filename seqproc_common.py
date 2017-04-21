#!/usr/bin/python
# seqproc_common.py - contains helper functions for the Sequence Processor


# imports
import numpy as np


# returns a feature engineered window (in the form of a list) given the video resolution and a list of jsons
def window (res_x, res_y, json_batch):


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
                # be careful: ymin and ymax are switched around by DeepDetect
                # code below is always correct, bug or not
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
            # write features to array
            features_base[frameno][itemno][:] = features


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


    # collect all engineered features
    output = features_base.flatten().tolist() + features_base_diff.flatten().tolist()
    # check that all values are normalized correctly
    assert(all(i >= -1 for i in output))
    assert(all(i <= 1 for i in output))
    # return window
    return output