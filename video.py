#!/usr/bin/python
# video.py - prepares video's for training the Sequence Processor

# input: a set of folders containing videoname.ext and videoname.txt in the same directory
# output: one folder per video, containing jpg-frames (both unannotated and annotated), json-ssd (from DeepDetect) and csv

# prerequisites:
# sudo apt-get install software-properties-common
# sudo add-apt-repository ppa:mc3man/trusty-media
# sudo apt-get update
# sudo apt-get install ffmpeg
# sudo apt-get install python-opencv
# sudo apt-get install python-imaging

# assumes DeepDetect Docker container is running, run following command on host OS:
# sudo nvidia-docker run -i -t -p 8080:8080 -v /home/ubuntu/dockershare:/dockershare:rshared beniz/deepdetect_gpu


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
import cv2
import re
import json
from dd_client import DD
from PIL import Image


# handle input arguments
parser = argparse.ArgumentParser(description='Process input data for training a Sequence Processor.')
parser.add_argument('builddir', help='build (timestamp only) that is to be tested')
parser.add_argument('-v', '--video', default='v', help='video that is to be processed')
parser.add_argument('-s', '--skipvids', default=False, action='store_true', help='do not extract the frames from the video again')
parser.add_argument('-i', '--iter', type=int, default=0, help='use a specific model iteration')
parser.add_argument('-f', '--framerate', type=float, default=1.0, help='how many frames to store and process per second')
parser.add_argument('-c', '--confidence-threshold', type=float, default=0.25, help='keep detections with confidence above threshold')
args = parser.parse_args()


# global pathing
folder_input = os.path.join(os.getcwd(), 'video', 'input')
folder_output = os.path.join(os.getcwd(), 'video', 'output')
if not os.path.exists(folder_output):
    os.makedirs(folder_output)


# gets the most recent iteration for a certain model build
def most_recent_iteration(build):
    files = os.listdir(os.path.join('builds', build, 'snapshots'))
    mostrecentiteration = 0
    for name in files:
        if name.lower().endswith('.caffemodel'):
            iteration = re.sub('\.caffemodel$', '', name)
            iteration = re.sub('ssd512x512_iter_', '', iteration)
            iteration = re.sub('ssd300x300_iter_', '', iteration)
            if (int(iteration) >= mostrecentiteration):
                mostrecentmodel = name
                mostrecentiteration = int(iteration)
    return mostrecentmodel


# build DeepDetect model repo
if not os.path.exists('dedemodel'):
    os.makedirs('dedemodel')
# copy static files
if not os.path.exists('dedemodel/deploy.prototxt'):
    shutil.copy2('includes/dede_deploy.prototxt', 'dedemodel/deploy.prototxt')
if not os.path.exists('dedemodel/corresp.txt'):
    shutil.copy2('includes/corresp.txt', 'dedemodel/corresp.txt')
# remove old models
for root, dirs, files in os.walk('dedemodel'):
    for name in files:
        if name.lower().endswith('.caffemodel'):
            os.remove(os.path.join(root, name))
# copy new model
recentmodel = most_recent_iteration(args.builddir)
print ('Using model ' + recentmodel)
shutil.copy2(os.path.join('builds', args.builddir, 'snapshots', recentmodel), 'dedemodel/model.caffemodel')


# setup DeepDetect service if necessary
dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)
model = {'repository':'/dockershare/ssd/dedemodel'}
parameters_input = {'connector':'image', 'width':512, 'height':512}
parameters_mllib = {'nclasses':7}
parameters_output = {}
detect = dd.delete_service('ssd')
detect = dd.put_service('ssd', model, 'single-shot detector', 'caffe', parameters_input, parameters_mllib, parameters_output, 'supervised')


# recursively process directory
for root, dirs, files in os.walk(folder_input):
    for name in files:
        name, ext = os.path.splitext(name)
        if (
            ext.lower().endswith(('.mp4', '.avi', '.mov')) and
            os.path.exists(os.path.join(root,name+'.txt')) and
            (args.video == 'v' or args.video == name)
        ):


            # start processing the video
            print ('Processing video '+name+'...')


            # video specific pathing
            output_jpg_unannotated = os.path.join(folder_output,'jpg_unannotated',name)
            if not os.path.exists(output_jpg_unannotated):
                os.makedirs(output_jpg_unannotated)
            output_jpg_annotated = os.path.join(folder_output,'jpg_annotated',name)
            if not os.path.exists(output_jpg_annotated):
                os.makedirs(output_jpg_annotated)
            output_json = os.path.join(folder_output,'json',name)
            if not os.path.exists(output_json):
                os.makedirs(output_json)
            folder_tags = os.path.join(folder_output,'tags')
            if not os.path.exists(folder_tags):
                os.makedirs(folder_tags)
            folder_resolution = os.path.join(folder_output,'resolution')
            if not os.path.exists(folder_resolution):
                os.makedirs(folder_resolution)
            output_tags = os.path.join(folder_tags,name+'.csv')
            output_resolution = os.path.join(folder_resolution,name+'.csv')


            # video -> jpg_unannotated
            if (args.skipvids == False):
                print ('  Converting video into unannotated frames...')
                cmd = 'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "'+root+'/'+name+ext+'"'
                duration = os.popen(cmd).read()
                i = 0.5
                while (i < float(duration)):
                    cmd = 'ffmpeg -y -nostats -loglevel 0 -accurate_seek -ss '+str(int(i)/3600).zfill(2)+':'+str(int(i)/60).zfill(2)+':'+str(int(i)%60).zfill(2)+'.5 -t 00:00:01 -i "'+root+'/'+name+ext+'" -r 1 -f singlejpeg "'+output_jpg_unannotated+'/'+name+'_'+str(int(i)+1).zfill(4)+'.jpg"'
                    os.system(cmd)
                    i = i + 1


            # jpg_unannotated -> res.csv
            image = Image.open(os.path.join(output_jpg_unannotated,name+'_0001.jpg'))
            with open(output_resolution, 'w+') as res:
                res.write(str(image.size[0])+','+str(image.size[1])+'\n')


            # txt -> csv
            print ('  Converting tagging txt into csv...')
            with open(os.path.join(root,name+'.txt')) as txt:
                with open(output_tags, 'w+') as csv:
                    # empty csv file
                    csv.truncate()
                    # original situation
                    minutes = 0
                    seconds = 0
                    status = 2
                    # process per line
                    for line in txt:
                        line = line.replace(' ', '')
                        if (len(line.strip()) != 0):
                            # translate human format to machine format
                            line = line.replace('nodig', '0')
                            line = line.replace('dig', '1')
                            line = line.replace(':', '')
                            # if the assertion fails, the tag file contains an error
                            assert len(line.strip()) == 5
                            # grab from line
                            prevminutes = minutes
                            prevseconds = seconds
                            prevstatus = status
                            minutes = line[0:2]
                            seconds = line[2:4]
                            status = line[4:5]
                            # calculate and push a new csv line, if not the first grab
                            if (prevstatus != 2):
                                pushstatus = prevstatus
                                pushlength = (int(seconds) + int(minutes) * 60) - (int(prevseconds) + int(prevminutes) * 60)
                                csv.write(str(pushlength)+','+str(pushstatus))
                                csv.write('\n')


            # jpg_unannotated -> json, jpg_annotated
            print ('  Processing unannotated frames through DeepDetect...')
            for subroot, subdirs, subfiles in os.walk(output_jpg_unannotated):
                for frame in sorted(subfiles):
                    parameters_input = {}
                    parameters_mllib = {'gpu':True}
                    parameters_output = {'bbox':True, 'confidence_threshold': args.confidence_threshold}
                    data = [os.path.join(output_jpg_unannotated,frame)]
                    detect = dd.post_predict('ssd',data,parameters_input,parameters_mllib,parameters_output)
                    #print detect
                    if detect['status']['code'] != 200:
                        print 'error',detect['status']['code']
                        sys.exit()
                    predictions = detect['body']['predictions']
                    with open(os.path.join(output_json,frame[:-4]+'.json'), 'w') as f:
                        json.dump(detect, f)
                        f.close()
                    for p in predictions:
                        img = cv2.imread(p['uri'])
                        # white image background, comment line below to see image behind boxes
                        cv2.rectangle(img,(0,9999),(9999,0),(255,255,255),-1)
                        for c in p['classes']:
                            cat = c['cat']
                            bbox = c['bbox']
                            if c['prob'] > args.confidence_threshold:
                                cv2.rectangle(img,(int(bbox['xmin']),int(bbox['ymax'])),(int(bbox['xmax']),int(bbox['ymin'])),(0,0,0),2)
                                cv2.putText(img,cat+' '+str("{0:.2f}".format(c['prob'])),(int(bbox['xmin']),int(bbox['ymax'])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
                        cv2.imwrite(os.path.join(output_jpg_annotated,frame[:-4]+'.jpg'),img)