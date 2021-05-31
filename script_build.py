from os import chdir
from subprocess import call,PIPE

chdir('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER/yolov4')
print('Building Model. Please Wait.')
call(['python','build_model.py'],stdin=PIPE,stdout=PIPE,stderr=PIPE)
print('Model built.')