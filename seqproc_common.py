#!/usr/bin/python
# seqproc_common.py - contains helper functions for the Sequence Processor

# prerequisites: sudo pip install sklearn


# imports
# no imports for now


# returns a feature engineered window (in the form of a list) given the video resolution and a list of jsons
def window (res_x, res_y, json_batch):
    # convert json frame sequence to window
    window_undifferenced = []
    window_size = 0
    number_of_features = 5
    # do for each frame
    for frame in json_batch:
        # up the window size
        window_size += 1
        # get the strongest detection for each category
        object_dict = {}
        # do for each object
        for detected_object in frame['body']['predictions'][0]['classes']:
            category = detected_object['cat']
            if category in object_dict:
                if object_dict[category]['prob'] < detected_object['prob']:
                    object_dict[category] = detected_object
            else:
                object_dict[category] = detected_object
        # present only excavator parts to the sequence processor
        ordering = ['cabin', 'forearm', 'upperarm', 'wheelbase', 'attachment-bucket', 'attachment-breaker']
        # write feature engineered window to array
        for item in ordering:
            if item in object_dict:
                # fetch json output and translate to relative positions
                # be careful: ymin and ymax are switched around by DeepDetect
                obj = object_dict[item]
                xmin = obj['bbox']['xmin'] / res_x
                xmax = obj['bbox']['xmax'] / res_x
                ymin = obj['bbox']['ymax'] / res_y
                ymax = obj['bbox']['ymin'] / res_y
                conf = obj['prob']
                # define features
                # also update number_of_features to avoid assertion fail
                C_X = (xmax - xmin)/2 + xmin
                C_Y = (ymax - ymin)/2 + ymin
                W = xmax - xmin
                H = ymax - ymin
                # extend window with features
                features = [C_X, C_Y, W, H, conf]
                assert(number_of_features == len(features))
                window_undifferenced.extend(features)
            else:
                # when an excavator part is not detected, extend with padding
                window_undifferenced.extend([0] * number_of_features)
    # difference each list item
    window_differenced = []
    for i in range(0, len(window_undifferenced)):
        # difference when not the first frame, otherwise, fill zeroes
        if i < number_of_features * len(ordering):
            window_differenced.extend([window_undifferenced[i], 0])
        else:
            window_differenced.extend([window_undifferenced[i], window_undifferenced[i] - window_undifferenced[i-(len(features)* len(ordering))]])
    # check the constructed window has the correct length
    assert(len(window_differenced) == number_of_features * 2 * len(ordering) * window_size)
    # return the constructed window
    return window_differenced