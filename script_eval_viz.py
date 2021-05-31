import os
import sys
os.chdir('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER')
sys.path.append('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER')
from visualizer import Visualizer

mot_img_path = '/content/drive/Shareddrives/ML: WHITE CHRISTMAS CENSOR/MOT16/train/MOT16-02/img1/'
image_seq = os.listdir(mot_img_path)
image_seq.sort()
input_image_sequence = [mot_img_path+_ for _ in image_seq]
out_seq = [[],[],[],[]]
init_seq = ['/content/eval_res/ev_und.txt','/content/eval_res/ev_fp.txt','/content/eval_res/ev_pr.txt','/content/eval_res/ev_gt.txt']
for img in image_seq:
  x = '/content/eval_viz_seq/'+img+'/'
  if not os.path.isdir(x):
    os.makedirs(x)
  out_seq[0].append(x+'undet.jpg')
  out_seq[1].append(x+'fpos.jpg')
  out_seq[2].append(x+'pr.jpg')
  out_seq[3].append(x+'gt.jpg')
viz = Visualizer()
for i in range(4):
  viz.INIT(init_seq[i],None)
  viz.RUN_ON_IMAGE_SEQUENCE(input_image_sequence, out_seq[i],None)