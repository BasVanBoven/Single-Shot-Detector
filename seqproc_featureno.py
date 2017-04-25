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
    ret = [ \
        'base feature cabin Cx', \
        'base feature cabin Cy', \
        'base feature cabin W', \
        'base feature cabin H', \
        'base feature cabin conf', \
        'base feature upperarm Cx', \
        'base feature upperarm Cy', \
        'base feature upperarm W', \
        'base feature upperarm H', \
        'base feature upperarm conf', \
        'base feature forearm Cx', \
        'base feature forearm Cy', \
        'base feature forearm W', \
        'base feature forearm H', \
        'base feature forearm conf', \
        'base feature wheelbase Cx', \
        'base feature wheelbase Cy', \
        'base feature wheelbase W', \
        'base feature wheelbase H', \
        'base feature wheelbase conf', \
        'base feature bucket Cx', \
        'base feature bucket Cy', \
        'base feature bucket W', \
        'base feature bucket H', \
        'base feature bucket conf', \
        'base feature breaker Cx', \
        'base feature breaker Cy', \
        'base feature breaker W', \
        'base feature breaker H', \
        'base feature breaker conf' \
    ]
    return ret[i%30]+' frame '+str((i/30)+1)


# helper function
def search_features_base_diff(i):
    ret = [ \
        'base feature difference cabin Cx', \
        'base feature difference cabin Cy', \
        'base feature difference cabin W', \
        'base feature difference cabin H', \
        'base feature difference cabin conf', \
        'base feature difference upperarm Cx', \
        'base feature difference upperarm Cy', \
        'base feature difference upperarm W', \
        'base feature difference upperarm H', \
        'base feature difference upperarm conf', \
        'base feature difference forearm Cx', \
        'base feature difference forearm Cy', \
        'base feature difference forearm W', \
        'base feature difference forearm H', \
        'base feature difference forearm conf', \
        'base feature difference wheelbase Cx', \
        'base feature difference wheelbase Cy', \
        'base feature difference wheelbase W', \
        'base feature difference wheelbase H', \
        'base feature difference wheelbase conf', \
        'base feature difference bucket Cx', \
        'base feature difference bucket Cy', \
        'base feature difference bucket W', \
        'base feature difference bucket H', \
        'base feature difference bucket conf', \
        'base feature difference breaker Cx', \
        'base feature difference breaker Cy', \
        'base feature difference breaker W', \
        'base feature difference breaker H', \
        'base feature difference breaker conf' \
    ]
    return ret[i%30]+' frame '+str((i/30)+1)


# helper function
def search_motility(i):
    ret = [ \
        'motility cabin', \
        'motility upperarm', \
        'motility forearm', \
        'motility wheelbase', \
        'motility bucket', \
        'motility breaker' \
    ]
    return ret[i]


# helper function
def search_relative_motility(i):
    return 'relative motility'


# helper function
def search_cabin_distance(i):
    ret = [ \
        'object cabin distance x+y upperarm', \
        'object cabin distance x upperarm', \
        'object cabin distance y upperarm', \
        'object cabin distance x+y forearm', \
        'object cabin distance x forearm', \
        'object cabin distance y forearm', \
        'object cabin distance x+y wheelbase', \
        'object cabin distance x wheelbase', \
        'object cabin distance y wheelbase', \
        'object cabin distance x+y bucket', \
        'object cabin distance x bucket', \
        'object cabin distance y bucket', \
        'object cabin distance x+y breaker', \
        'object cabin distance x breaker', \
        'object cabin distance y breaker' \
    ]
    return ret[i%15]+' frame '+str((i/15)+1)


# helper function
def search_breaker_vs_bucket(i):
    ret = [ \
        'breaker vs bucket conf difference', \
        'breaker vs bucket forearm distance x+y', \
        'breaker vs bucket forarm distance x', \
        'breaker vs bucket forarm distance y' \
    ]
    return ret[i]


# helper function
def search_object_size(i):
    ret = [ \
        'object size x+y cabin', \
        'object size x cabin', \
        'object size y cabin', \
        'object size x+y upperarm', \
        'object size x upperarm', \
        'object size y upperarm', \
        'object size x+y forearm', \
        'object size x forearm', \
        'object size y forearm', \
        'object size x+y wheelbase', \
        'object size x wheelbase', \
        'object size y wheelbase', \
        'object size x+y bucket', \
        'object size x bucket', \
        'object size y bucket', \
        'object size x+y breaker', \
        'object size x breaker', \
        'object size y breaker' \
    ]
    return ret[i]


# helper function
def search_object_size_relative(i):
    ret = [ \
        'relative object size x+y', \
        'relative object size x', \
        'relative object size y' \
    ]
    return ret[i]


# helper function
def search_object_size_warping(i):
    ret = [ \
        'object size warping x+y cabin', \
        'object size warping x cabin', \
        'object size warping y cabin', \
        'object size warping x+y upperarm', \
        'object size warping x upperarm', \
        'object size warping y upperarm', \
        'object size warping x+y forearm', \
        'object size warping x forearm', \
        'object size warping y forearm', \
        'object size warping x+y wheelbase', \
        'object size warping x wheelbase', \
        'object size warping y wheelbase', \
        'object size warping x+y bucket', \
        'object size warping x bucket', \
        'object size warping y bucket', \
        'object size warping x+y breaker', \
        'object size warping x breaker', \
        'object size warping y breaker' \
    ]
    return ret[i]