#!/usr/bin/python
# setup.py - processes input data for training a Single Shot Detector

# input: a frame folder containing jpg (arbitrary size) and xml files
# to get caffe in your python path, execute 'export PYTHONPATH=/caffe/python:$PYTHONPATH'


# imports
import os
os.environ['GLOG_minloglevel'] = '2'
import argparse
import math
import shutil
import stat
import subprocess
import sys
import random
import PIL
import time
import datetime
import random
import caffe
import xml.etree.ElementTree as ET
from caffe.proto import caffe_pb2
from caffe.model_libs import *
from google.protobuf import text_format
from shutil import copyfile
from shutil import copytree
from random import shuffle
from PIL import Image


# handle input arguments
parser = argparse.ArgumentParser(description='Process input data for training a Single Shot Detector.')
parser.add_argument('-l', '--large', default=False, action='store_true', help='use the SSD512 architecture')
parser.add_argument('-s', '--stop', default=False, action='store_true', help='do not start training after setup')
parser.add_argument('-t', '--test', type=float, default=0.2, help='percentage of images in test set')
args = parser.parse_args()


# global parameters
# current working directory
rootdir = os.getcwd()
# caffe root directory
rootcaffe = '/caffe'
# folder which contains images and labels
sourcedir = 'frames'
# percentage of images in test set
testratio = args.test
# longest edge of image, usually 300 or 512 pixels
if args.large == True:
    resize = 512
else:
    resize = 300


# create build directory and subdirectories
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
os.makedirs(os.path.join('builds', timestamp))
os.makedirs(os.path.join('builds', timestamp, 'trainval', 'image'))
os.makedirs(os.path.join('builds', timestamp, 'trainval', 'label'))
os.makedirs(os.path.join('builds', timestamp, 'test', 'image'))
os.makedirs(os.path.join('builds', timestamp, 'test', 'label'))
copytree(os.path.join(rootdir, 'includes'), os.path.join(rootdir, 'builds', timestamp, 'includes'))


# copy data to trainval folder, replacing spaces in filenames with underscores
datacount = 0
for root, dirs, files in os.walk(sourcedir):
    for name in files:
        name, ext = os.path.splitext(name)
        if (ext.lower() == '.jpg' and os.path.exists(os.path.join(root, name + '.xml'))):
            datacount += 1
            assert os.path.exists(os.path.join('builds', timestamp, 'trainval', 'image', name + ext)) == False
            assert os.path.exists(os.path.join('builds', timestamp, 'trainval', 'label', name + '.xml')) == False
            copyfile(os.path.join(root, name + ext), os.path.join('builds', timestamp, 'trainval', 'image', name.replace(" ", "_") + ext))
            copyfile(os.path.join(root, name + '.xml'), os.path.join('builds', timestamp, 'trainval', 'label', name.replace(" ", "_") + '.xml'))
print 'moved ' + str(datacount) + ' images (with annotations) to trainval folder'


# resize all images and labels
print 'resizing all images (and annotations)...'
files = os.listdir(os.path.join('builds', timestamp, 'trainval', 'label'))
for name in files:
    tree = ET.parse(os.path.join('builds', timestamp, 'trainval', 'label', name))
    root = tree.getroot()
    oldwidth = root.find('size').find('width').text
    oldheight = root.find('size').find('height').text
    if (int(oldwidth) > int(oldheight)):
        crop = resize / float(oldwidth)
    else:
        crop = resize / float(oldheight)
    for size in root.iter('size'):
        size.find('width').text = str(int(int(oldwidth) * crop))
        size.find('height').text = str(int(int(oldheight) * crop))
    for bndbox in root.iter('bndbox'):
        bndbox.find('xmin').text = str(int(float(bndbox.find('xmin').text) * crop))
        bndbox.find('xmax').text = str(int(float(bndbox.find('xmax').text) * crop))
        bndbox.find('ymin').text = str(int(float(bndbox.find('ymin').text) * crop))
        bndbox.find('ymax').text = str(int(float(bndbox.find('ymax').text) * crop))
    tree.write(os.path.join('builds', timestamp, 'trainval', 'label', name))
    name, ext = os.path.splitext(name)
    image = Image.open(os.path.join('builds', timestamp, 'trainval', 'image', name + '.jpg'))
    maxsize = (resize, resize)
    image = image.resize((int(int(oldwidth) * crop), int(int(oldheight) * crop)), Image.ANTIALIAS)
    image.save(os.path.join('builds', timestamp, 'trainval', 'image', name + '.jpg'))
print 'done resizing all images (and annotations)'


# copy a fixed percentage of random images and labels to test folder
tomove = datacount * testratio
files = os.listdir(os.path.join('builds', timestamp, 'trainval', 'image'))
files = random.sample(files, int(tomove))
for name in files:
    name, ext = os.path.splitext(name)
    shutil.move(os.path.join('builds', timestamp, 'trainval', 'image', name + ext), os.path.join('builds', timestamp, 'test', 'image', name + ext))
    shutil.move(os.path.join('builds', timestamp, 'trainval', 'label', name + '.xml'), os.path.join('builds', timestamp, 'test', 'label', name + '.xml'))
print 'moved ' + str(int(tomove)) + ' images (with annotations) to test folder'


# generate trainval.txt, test.txt and test_name_size.txt (trainval_name_size.txt is not used)
for imageset in ['trainval', 'test']:
    output = open(os.path.join('builds', timestamp, imageset + '.txt'),'w')
    output_size = open(os.path.join('builds', timestamp, imageset + '_name_size.txt'),'w')
    for root, dirs, files in os.walk(os.path.join('builds', timestamp, imageset, 'image')):
        for name in sorted(files):
            name, ext = os.path.splitext(name)
            output.write('/'+os.path.join('builds', timestamp, imageset, 'image', name + '.jpg') + ' ' + '/'+os.path.join('builds', timestamp, imageset, 'label', name + '.xml') + os.linesep)
            image = Image.open(os.path.join('builds', timestamp, imageset, 'image',  name + '.jpg'))
            width, height = image.size
            output_size.write(name + ' ' + str(width) + ' ' + str(height) + os.linesep)
    output.close()
    output_size.close()
print 'generated trainval.txt, test.txt and test_name_size.txt in the ' + timestamp + ' build directory'


# shuffle trainval.txt and test.txt
for imageset in ['trainval', 'test']:
    with open(os.path.join('builds', timestamp, imageset + '.txt'),'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(os.path.join('builds', timestamp, imageset + '.txt'),'w') as target:
        for _, line in data:
            target.write(line)


# generate LMDB
for imageset in ['trainval', 'test']:
    cmd = rootcaffe+'/build/tools/convert_annoset' \
        ' --anno_type=detection' \
        ' --label_type=xml' \
        ' --label_map_file='+rootdir+'/includes/labelmap.prototxt' \
        ' --check_label=true' \
        ' --min_dim=0' \
        ' --max_dim=0' \
        ' --resize_height=0' \
        ' --resize_width=0' \
        ' --backend=lmdb' \
        ' --shuffle=false' \
        ' --check_size=false' \
        ' --encode_type=jpg' \
        ' --encoded=true' \
        ' --gray=false' \
        ' {} {} {}'.format(rootdir, os.path.join(rootdir, 'builds', timestamp, imageset+'.txt'), os.path.join(rootdir, 'builds', timestamp, 'lmdb_'+imageset))
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
print 'generated lmdb files'


# save ssd file
output = open(os.path.join('builds', timestamp, 'ssd'+str(resize)+'.log'),'w')
output.write('This model uses the SSD'+str(resize)+' architecture.')
output.close()


# start training
if args.stop == False:
    os.system('python '+rootdir+'/train.py '+timestamp)