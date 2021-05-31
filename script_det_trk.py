import os
os.chdir('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER/')
import sys
sys.path.append('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER/')
from object_tracker import ObjectTracker
import time

print('Initializing Object Detector/Tracker')
start_time = time.time()

objt = ObjectTracker()
objt.INIT_OBJECT_TRACKER()

end_time = time.time()
print(f'Object Detector/Tracker Initialized. Time Elapsed: {end_time-start_time:.2f}')
print()

#objt.SET_FLAG('allowed_classes',['bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'snowboard',  'skateboard', 'surfboard'])
objt.SET_FLAG('allowed_classes',['person'])
mot_img_path = '/content/drive/Shareddrives/ML: WHITE CHRISTMAS CENSOR/MOT16/train/MOT16-02/img1/'
img_seq = [mot_img_path+_ for _ in os.listdir(mot_img_path)]
img_seq.sort()
objt.RUN_ON_IMAGE_SEQUENCE(img_seq,'/content/det.txt','/content/trk.txt')

#objt.RUN_ON_VIDEO('/content/input.mp4','/content/det2.txt','/content/trk2.txt')