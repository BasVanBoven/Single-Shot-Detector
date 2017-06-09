#!/usr/bin/python
# buildlmdb.py - processes input data for a specific test set

# input: a frame folder containing jpg (arbitrary size) and xml files
# to get caffe in your python path, execute 'export PYTHONPATH=/caffe/python:$PYTHONPATH'


# imports
import argparse
import math
import os
import shutil
import stat
import subprocess
import sys
import random
import PIL
import time
import datetime
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
parser = argparse.ArgumentParser(description='Process input data for a specific test set.')
parser.add_argument('-l', '--large', default=True, action='store_true', help='use the SSD512 architecture')
args = parser.parse_args()


# global parameters
# current working directory
rootdir = os.getcwd()
# caffe root directory
rootcaffe = '/caffe'
# folder which contains images and labels
sourcedir = os.path.join(rootdir, 'testsets_benchmark', 'crawl')
# longest edge of image, usually 300 or 512 pixels
if args.large == True:
    resize = 512
else:
    resize = 300


# create build directory and subdirectories
shutil.rmtree(os.path.join(rootdir, 'builds', 'crawl'))
os.makedirs(os.path.join(rootdir, 'builds', 'crawl', 'test', 'image'))
os.makedirs(os.path.join(rootdir, 'builds', 'crawl', 'test', 'label'))


# copy data to trainval folder, replacing spaces in filenames with underscores
datacount = 0
for root, dirs, files in os.walk(sourcedir):
    random.shuffle(files)
    for name in files:
        name, ext = os.path.splitext(name)
        if (ext.lower() == '.jpg' and os.path.exists(os.path.join(root, name + '.xml'))):
            datacount += 1
            copyfile(os.path.join(root, name + ext), os.path.join('builds', 'crawl', 'test', 'image', name.replace(" ", "_") + ext))
            copyfile(os.path.join(root, name + '.xml'), os.path.join('builds', 'crawl', 'test', 'label', name.replace(" ", "_") + '.xml'))
print 'moved ' + str(datacount) + ' images (with annotations) to test folder'


# resize all images and labels
print 'resizing all images (and annotations)...'
files = os.listdir(os.path.join('builds', 'crawl', 'test', 'label'))
for name in files:
    tree = ET.parse(os.path.join('builds', 'crawl', 'test', 'label', name))
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
    tree.write(os.path.join('builds', 'crawl', 'test', 'label', name))
    name, ext = os.path.splitext(name)
    image = Image.open(os.path.join('builds', 'crawl', 'test', 'image', name + '.jpg'))
    maxsize = (resize, resize)
    image = image.resize((int(int(oldwidth) * crop), int(int(oldheight) * crop)), Image.ANTIALIAS)
    image.save(os.path.join('builds', 'crawl', 'test', 'image', name + '.jpg'))
print 'done resizing all images (and annotations)'


# generate test.txt and test_name_size.txt (trainval_name_size.txt is not used)
output = open(os.path.join('builds', 'crawl', 'test' + '.txt'),'w')
output_size = open(os.path.join('builds', 'crawl', 'test' + '_name_size.txt'),'w')
for root, dirs, files in os.walk(os.path.join('builds', 'crawl', 'test', 'image')):
    for name in sorted(files):
        name, ext = os.path.splitext(name)
        output.write('/'+os.path.join('builds', 'crawl', 'test', 'image', name + '.jpg') + ' ' + '/'+os.path.join('builds', 'crawl', 'test', 'label', name + '.xml') + os.linesep)
        image = Image.open(os.path.join('builds', 'crawl', 'test', 'image',  name + '.jpg'))
        width, height = image.size
        output_size.write(name + ' ' + str(width) + ' ' + str(height) + os.linesep)
output.close()
output_size.close()
print 'generated test.txt and test_name_size.txt in the crawl directory'


# shuffle trainval.txt and test.txt
with open(os.path.join('builds', 'crawl', 'test' + '.txt'),'r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open(os.path.join('builds', 'crawl', 'test' + '.txt'),'w') as target:
    for _, line in data:
        target.write(line)


# generate LMDB
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
    ' {} {} {}'.format(rootdir, os.path.join(rootdir, 'builds', 'crawl', 'test'+'.txt'), os.path.join(rootdir, 'builds', 'crawl', 'lmdb_'+'test'))
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output = process.communicate()[0]
print 'generated lmdb file'