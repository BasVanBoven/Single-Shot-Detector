import os, sys, argparse
from dd_client import DD
import cv2
import json

parser = argparse.ArgumentParser()
parser.add_argument("--image",help="path to image")
parser.add_argument("--confidence-threshold",help="keep detections with confidence above threshold",type=float,default=0.1)
args = parser.parse_args()

host = 'localhost'
sname = 'ssd512'
description = 'image classification'
mllib = 'caffe'
mltype = 'supervised'
nclasses = 9
width = height = 512
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)

# creating ML service
model_repo = '/dockershare/ssd_detect_512' + '/model'
model = {'repository':model_repo}
parameters_input = {'connector':'image','width':width,'height':height}
parameters_mllib = {'nclasses':nclasses}
parameters_output = {}
detect = dd.put_service(sname,model,description,mllib,
               parameters_input,parameters_mllib,parameters_output,mltype)
print detect

# prediction call
frames = os.listdir('/dockershare/ssd_detect_512/frames')
for frame in sorted(frames):
    parameters_input = {}
    parameters_mllib = {'gpu':True}
    parameters_output = {'bbox':True, 'confidence_threshold': args.confidence_threshold}
    data = ['/dockershare/ssd_detect_512/frames/'+frame]
    detect = dd.post_predict(sname,data,parameters_input,parameters_mllib,parameters_output)
    print detect
    if detect['status']['code'] != 200:
        print 'error',detect['status']['code']
        sys.exit()
    predictions = detect['body']['predictions']
    with open('/dockershare/ssd_detect_512/txt/'+frame[:-4]+'.txt', 'w') as f:
        json.dump(detect, f)
        f.close()
    for p in predictions:
        img = cv2.imread('/'+p['uri'])
        for c in p['classes']:
            cat = c['cat']
            bbox = c['bbox']
            if c['prob'] > 0.35:
                cv2.rectangle(img,(int(bbox['xmin']),int(bbox['ymax'])),(int(bbox['xmax']),int(bbox['ymin'])),(255,0,0),2)
                cv2.putText(img,cat,(int(bbox['xmin']),int(bbox['ymax'])),cv2.FONT_HERSHEY_PLAIN,1,255)
        cv2.imwrite('/dockershare/ssd_detect_512/annotations/'+frame[:-4]+'.jpg',img)