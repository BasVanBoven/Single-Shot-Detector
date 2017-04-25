#!/usr/bin/python
# seqproc_featureno.py - contains helper functions for the Sequence Processor


# imports
import numpy as np


# main function: routs through to array-specific function
# i = index, w = windowsize, o = objects, f = features, l = length
def featureno(i, w):

    o = 6
    f = 5
    l = [ w*o*f, (w-1)*o*f, o, 1, w*(o-1)*3, 4, o*3, 3, o*3 ]
    l = np.cumsum(l)

    if i in range(0, l[0]):
        return search_features_base(i-0)
    elif i in range(l[0], l[1]):
        return search_features_base_diff(i-l[0])
    elif i in range(l[1], l[2]):
        return search_motility(i-l[1])
    elif i in range(l[2], l[3]):
        return search_relative_motility(i-l[2])
    elif i in range(l[3], l[4]):
        return search_cabin_distance(i-l[3])
    elif i in range(l[4], l[5]):
        return search_breaker_vs_bucket(i-l[4])
    elif i in range(l[5], l[6]):
        return search_object_size(i-l[5])
    elif i in range(l[6], l[7]):
        return search_object_size_relative(i-l[6])
    elif i in range(l[7], l[8]):
        return search_object_size_warping(i-l[7])
    else:
        return 'Range error!'


# helper function
def search_features_base(i):
    return 'features_base'

# helper function
def search_features_base_diff(i):
    return 'features_base_diff'

# helper function
def search_motility(i):
    return 'motility'

# helper function
def search_relative_motility(i):
    return 'relative_motility'

# helper function
def search_cabin_distance(i):
    return 'cabin_distance'

# helper function
def search_breaker_vs_bucket(i):
    return 'breaker_vs_bucket'

# helper function
def search_object_size(i):
    return 'object_size'

# helper function
def search_object_size_relative(i):
    return 'object_size_relative'

# helper function
def search_object_size_warping(i):
    return 'object_size_warping'