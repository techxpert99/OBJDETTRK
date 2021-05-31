import os
import sys

os.chdir('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER')
sys.path.append('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER')

from visualizer import Visualizer

viz = Visualizer()
viz.INIT('/content/det.txt','/content/trk.txt')

mot_img_path = '/content/drive/Shareddrives/ML: WHITE CHRISTMAS CENSOR/MOT16/train/MOT16-02/img1/'
image_seq = os.listdir(mot_img_path)
image_seq.sort()
input_image_sequence = [mot_img_path+_ for _ in image_seq]
output_image_detect_sequence = ['/content/detect_seq/'+_ for _ in image_seq]
output_image_track_sequence = ['/content/track_seq/'+_ for _ in image_seq]

if not os.path.isdir('/content/detect_seq'):
  os.mkdir('/content/detect_seq')
if not os.path.isdir('/content/track_seq'):
  os.mkdir('/content/track_seq')

#viz.RUN_ON_IMAGE('/content/drive/Shareddrives/ML: WHITE CHRISTMAS CENSOR/MOT16/train/MOT16-02/img1/000001.jpg','/content/viz_gt.jpg','/content/viz_trk.jpg',1)

viz.RUN_ON_IMAGE_SEQUENCE(input_image_sequence,output_image_detect_sequence,output_image_track_sequence)

#viz.RUN_ON_VIDEO('/content/input.mp4','/content/detect.mp4','/content/track.mp4')