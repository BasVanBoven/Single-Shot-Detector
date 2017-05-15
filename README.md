# SPEAR Model Tools


## Introduction

This repository contains all the code necessary to train, test and benchmark two models. The first model is an real-time object detection model, based on an implementation of the [Single-Shot Detector](https://arxiv.org/abs/1512.02325) in Caffe. The second model is a action classification model, which takes bounding boxes that the Single-Shot Detector generates and is able to classify based on this information. All of the code was written by [Bas van Boven](http://basvanboven.nl) during his research internship at Accenture in 2017, with the exception of `train.py`, which is heavily inspired by the [original Single-Shot Detector code](https://github.com/weiliu89/caffe/tree/ssd) by Wei Liu.

The big benefit of our object and action detection framework versus other methods is that our method is insensitive to unrelated objects and movement within the frame.

The code in this repository is geared towards the original project, but is modular enough to fit similar use cases without much additional coding. For more information about the original project, please have a look at the original [Master Thesis](http://basvanboven.nl/transfer/thesis.pdf).


## Data Prerequisites

The Model Tools will need a dataset that corresponds to the use case at hand for both the Object Detector and Action Classifier.

The Object Detector assumes a folder `frames` in the root of this repository, which should contain images (jpg) and corresponding labeldata (xml) with the same name. The `frame` folder is scanned recursively, so you can adhere to your own file structure within this folder. Also, to make your life easier, you can find three useful utilities in the `utilities` folder:

- `frameextractor.py`: uses ffmpeg to extract frames from video files (can be run locally in a Windows or Linux environment as long as Python 2.7 is installed).
- `labelimg`: tool with a GUI that you can run to generate labeldata for images. Can be compiled to both a Windows and Linux version.
- `list.py`: displays the number of frames per folder, useful for reporting on the PM side.

If you use shared cloud storage between your project, you can distribute the tagging work rather easily.

The Action Classifier assumes a folder `video/input` in the root of this repository, which should contain video files (mp4, mov or avi) and corresponding labeldata (txt) with the same name. The labeling format is custom and can be defined as a list of `<min>:<sec> <classification>`, where each list item is on a new line. Thus, the current setup does not support video's longer than an hour. Do not worry about line breaks too much in this setup.


## Operating Environment

Originally, we ran the scripts on an Amazon instance within Docker containers. For training and development, you can use any Caffe Docker, although we built a custom image for the original project.

For serving the Object Detector model, you can use the [Deepdetect GPU Image](https://hub.docker.com/r/beniz/deepdetect_gpu/), or the [DeepDetect Pascal Image](https://hub.docker.com/r/beniz/deepdetect_gpu_pascal/) if you want to serve on a GPU with Pascal architecture.

Serving the Action Classifier requires only a few simple Python packages, so you have a few options here. We recommend [Anaconda](https://hub.docker.com/r/continuumio/anaconda/). The scripts are optimized for a `p2.xlarge` instance, but they can be configured to run on any instance as long as it has a GPU (for training the Object Detector). Please be aware that this is a resource-intensive process and will likely take multiple days.


## Object Detector (Single-Shot Detector)

- `frameinspector.py`: finds duplicate frames and frames without tags.
- `setup.py`: processes input data for training a Single Shot Detector.
- `train.py`: starts training a Single Shot Detector.
- `test.py`: tests a Single Shot Detector, returns human-viewable tagged images.
- `benchmark.py`: benchmarks a Single Shot Detector on a variety of self-constructed test sets (in `testsets` folder)


## Action Classifier (Sequence Processor)

- `video.py`: prepares videos for training the Action Classifier, may take several hours.
- `seqproc_setup.py`: processes input data for training an Action Classifier.
- `seqproc_train.py`: processes input data for training an Action Classifier.
- `seqproc_serv.py`: hosts a movement detector model over API via flask.
- `seqproc_test.py`: sends test requests to the sequence_processor API.
- `seqproc_3dplot.py`: can be used for parameter optimisation. In most cases, AdaBoost with a window size of 3 is only outperformed by AdaBoost with a larger window size, which increases latency.