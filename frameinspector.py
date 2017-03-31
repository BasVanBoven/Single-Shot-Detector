#!/usr/bin/python
# frameinspector.py - finds duplicate frames and frames without tags

# input: a frame folder containing jpg (arbitrary size) and xml files


# imports
import argparse
import math
import os
import shutil
import stat
import subprocess
import sys
import random
import time
import datetime
import random
import xml.etree.ElementTree as ET
from shutil import copyfile
from shutil import copytree
from random import shuffle


# handle input arguments
parser = argparse.ArgumentParser(description='Process input data for training a Single Shot Detector.')
parser.add_argument('sourcedir', help='directory that is to be processed')
parser.add_argument('-p', '--pretend', default=False, action='store_true', help='disable write mode')
args = parser.parse_args()


# returns true if all bounding boxes are equal, however, earth rod images are never equal
def equal_bbox(current_tree, previous_tree):
    previous_root = previous_tree.getroot()
    current_root = current_tree.getroot()
    for original in current_root.findall('object'):
        # frame is not a duplicate if it has an earth rod in it
        if (original.find('name').text == 'manual-earthrod'):
            return False
        # do duplicate check
        found = False
        for match in previous_root.findall('object'):
            if (
                original.find('name').text == match.find('name').text and
                original.find('bndbox').find('xmin').text == match.find('bndbox').find('xmin').text and
                original.find('bndbox').find('xmax').text == match.find('bndbox').find('xmax').text and
                original.find('bndbox').find('ymin').text == match.find('bndbox').find('ymin').text and
                original.find('bndbox').find('ymax').text == match.find('bndbox').find('ymax').text
            ):
                found = True
        if found == False:
            # no similar object has been found for the original: frame is not a duplicate
            return False
    # all originals have a match: frame is a duplicate
    return True


# reset counters
datacount = 0
count_missinglabels = 0
count_similar = 0
count_cabin = 0
count_wheelbase = 0
count_forearm = 0
count_upperarm = 0
count_attachment_bucket = 0
count_attachment_breaker = 0
count_manual_crouching = 0
count_manual_earthrod = 0


# warn if running without pretend flag
if (args.pretend == False):
    answer = raw_input('Warning: running without the pretend flag. Continue (y/n)? ')
    if (answer.lower() != 'y'):
        exit()


# walk through whole folder
for root, dirs, files in sorted(os.walk(args.sourcedir)):
    for name in sorted(files):
        name, ext = os.path.splitext(name)
        if ext.lower() == '.jpg':
            datacount += 1
            if os.path.exists(os.path.join(root, name + '.xml')) == False:
                count_missinglabels += 1
                print ('Image ' + os.path.join(root, name + '.jpg') + ' has no labels')
                if (args.pretend == False):
                    os.remove(os.path.join(root, name + '.jpg'))
            else:
                current_tree = ET.parse(os.path.join(root, name + '.xml'))
                if (datacount > 1 and equal_bbox(current_tree, previous_tree)):
                    count_similar += 1
                    print ('Image ' + os.path.join(root, name + '.jpg') + ' is similar to the previous frame')
                    if (args.pretend == False):
                        os.remove(os.path.join(root, name + '.jpg'))
                        os.remove(os.path.join(root, name + '.xml'))
                else:
                    # not similar
                    current_root = current_tree.getroot()
                    for object in current_root.findall('object'):
                        if object.find('name').text == 'cabin':
                            count_cabin += 1
                        elif object.find('name').text == 'wheelbase':
                            count_wheelbase += 1
                        elif object.find('name').text == 'forearm':
                            count_forearm += 1
                        elif object.find('name').text == 'upperarm':
                            count_upperarm += 1
                        elif object.find('name').text == 'attachment-bucket':
                            count_attachment_bucket += 1
                        elif object.find('name').text == 'attachment-breaker':
                            count_attachment_breaker += 1
                        elif object.find('name').text == 'manual-crouching':
                            count_manual_crouching += 1
                        elif object.find('name').text == 'manual-earthrod':
                            count_manual_earthrod += 1
        previous_tree = current_tree


# print statistics
print ''
print ''
print 'Scanned ' + str(datacount) + ' images:'
print str(count_missinglabels) + ' missing labels'
print str(count_similar) + ' similar frames'
if (args.pretend):
    print 'Run without the pretend flag (-p) to remove these images'
else:
    print 'These images and corresponding labels were removed'
print ''
print '(Usable) tag count:'
print str(count_cabin) + ' cabin'
print str(count_wheelbase) + ' wheelbase'
print str(count_forearm) + ' forearm'
print str(count_upperarm) + ' upperarm'
print str(count_attachment_bucket) + ' attachment-bucket'
print str(count_attachment_breaker) + ' attachment-breaker'
print str(count_manual_crouching) + ' manual-crouching'
print str(count_manual_earthrod) + ' manual-earthrod'