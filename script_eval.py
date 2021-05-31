import os
import sys
os.chdir('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER')
sys.path.append('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER')

from evaluator import Evaluator

eval = Evaluator()
if not os.path.isdir('/content/eval_res/'):
  os.mkdir('/content/eval_res/')
eval_res = ['/content/eval_res/ev_und.txt','/content/eval_res/ev_fp.txt','/content/eval_res/ev_pr.txt','/content/eval_res/ev_gt.txt']
eval.INIT('/content/drive/Shareddrives/ML: WHITE CHRISTMAS CENSOR/MOT16/train/MOT16-02/gt/gt.txt','mot-16','/content/det.txt','/content/trk.txt',eval_res)

eval.EVALUATE_DETECTIONS()