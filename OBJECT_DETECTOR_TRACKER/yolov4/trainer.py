import cv2
import utils
import os
import numpy as np
import tensorflow as tf

FLAGS = dict()

def LOAD_FLAGS():
  global FLAGS
  config = utils.READ_CONFIG_FILE(utils.CONFIG_FILENAME)
  FLAGS['dataset_path'] = config['TRAIN.DATASET_PATH']
  FLAGS['batch_size'] = utils.PARSE_ITEM(config['TRAIN.BATCH_SIZE'],int)
  FLAGS['input_size'] = utils.PARSE_ITEM(config['MODEL.INPUT_SIZE'],int)
  FLAGS['learn_rate_init'] = utils.PARSE_ITEM(config['TRAIN.LEARN_RATE_INIT'],float)
  FLAGS['learn_rate_end'] = utils.PARSE_ITEM(config['TRAIN.LEARN_RATE_END'],float)
  FLAGS['num_epochs'] = utils.PARSE_ITEM(config['TRAIN.NUM_EPOCHS'],int)
  FLAGS['validation_split'] = utils.PARSE_ITEM(config['TRAIN.VALIDATION_SPLIT'],float)
  FLAGS['num_workers'] = utils.PARSE_ITEM(config['TRAIN.NUM_WORKERS'],int)
  FLAGS['multiprocessing'] = config['TRAIN.MULTIPROCESSING'] == 'True'
  FLAGS['buffer_size'] = utils.PARSE_ITEM(config['TRAIN.BUFFER_SIZE'],int)
  FLAGS['dataset_type'] = config['TRAIN.DATASET'].lower()
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.LOAD_CONFIG()
  FLAGS['strides'] = STRIDES
  FLAGS['anchor_per_scale'] = utils.PARSE_ITEM(config['YOLO.ANCHOR_PER_SCALE'],int)
  FLAGS['num_classes'] = NUM_CLASS
  FLAGS['max_bbox_per_scale'] = utils.PARSE_ITEM(config['YOLO.MAX_BBOX_PER_SCALE'],int)
  FLAGS['anchors'] = ANCHORS
  FLAGS['iou_loss_threshold'] = utils.PARSE_ITEM(config['YOLO.IOU_LOSS_THRESH'],float)
  RESET_BUFFER()

def CONCAT_PATH(parent,child):
  if parent[-1] in ['/','\\']: return parent+child
  return parent+'/'+child

def PARENT_PATH(path):
  for i in range(len(path)-1,-1,-1):
    if path[i] in ['/','\\']:
      return path[:i]
  return None

def EXPLORE_DIR(direc,maxlevels=-1):
  if os.path.isfile(direc) or (maxlevels==0 and os.path.isdir(direc)):
    return [direc]
  elif os.path.isdir(direc) and maxlevels != 0:
    files = []
    for fil in os.listdir(direc):
      abs_fil = CONCAT_PATH(direc,fil)
      files.extend(EXPLORE_DIR(abs_fil,maxlevels-1))
    return files
  return []

def READ_LABELS(label_file,image_path):
  dtype = FLAGS['dataset_type']
  labeldict = dict()
  if dtype == 'mot-16':
    with open(label_file,'r') as k:
      for line in k.read().splitlines():
        tmp = utils.PARSE_LIST(line,float)
        fn,xmin,ymin,w,h,con,cls = int(tmp[0]),tmp[2],tmp[3],tmp[2]+tmp[4],tmp[3]+tmp[5],tmp[6],tmp[7]
        fname = CONCAT_PATH(image_path,('0'*(6-len(line[0]))+str(fn)+'.jpg'))
        if fname not in labeldict:
          labeldict[fname] = set()
        if con:
          labeldict[fname].add((xmin,ymin,w,h,cls))
  FLAGS['label_dict'] = labeldict

def DISCOVER_DATASET():
  dtype = FLAGS['dataset_type']
  dataset = list()
  if dtype == 'mot-16':
    dpath = FLAGS['dataset_path']
    dirs = EXPLORE_DIR(dpath,1)
    for f in dirs:
      image_path = CONCAT_PATH(f,'img1')
      lab_file = CONCAT_PATH(f,'gt/gt.txt')
      READ_LABELS(lab_file,image_path)
      labels = FLAGS['label_dict']   
      image_files = EXPLORE_DIR(image_path,1)
      for abs_image in image_files:
        if DETECT_FILETYPE(abs_image) == 'image':
          if abs_image in labels:
            dataset.append((abs_image,labels[abs_image]))
      del FLAGS['label_dict']
  FLAGS['files'] = dataset

def RESET_BUFFER():
  FLAGS['buf_index'] = 0
  FLAGS['buf_vid_index'] = 0

def LOAD_BUFFER():
  buf = list()
  loaded = 0
  while FLAGS['buf_index'] < len(FLAGS['files']) and loaded < FLAGS['buffer_size']:
    fname,info = FLAGS['files'][FLAGS['buf_index']]
    ftype = DETECT_FILETYPE(fname)
    if ftype == 'image':
      frame = LOAD_FRAME_FROM_IMAGE(fname)
      if frame is not None:
        buf.append((frame,info))
        loaded += 1
      FLAGS['buf_index'] += 1
    elif ftype == 'video':
      rem = FLAGS['buffer_size']-loaded
      frames = LOAD_FRAME_FROM_VIDEO(fname,FLAGS['buf_vid_index'],rem)
      frames = [(frames[i],info[FLAGS['buf_vid_index']+i]) for i in range(len(frames))]
      if len(frames) < rem:
        FLAGS['buf_vid_index'] = 0
        FLAGS['buf_index'] += 1
      else:
        FLAGS['buf_vid_index'] += len(frames)
      loaded += len(frames)
      buf.extend(frames)
    else:
      FLAGS['buf_index'] += 1
      FLAGS['buf_vid_index'] = 0
  FLAGS['buffer'] = buf

def DATASET_HAS_MORE():
  return FLAGS['buf_index'] < len(FLAGS['files'])

def DETECT_FILETYPE(filename):
  extension = filename.split('.')[-1]
  if extension in ['jpg','png']: return 'image'
  elif extension in ['mp4','avi','mkv']: return 'video'
  return None

def LOAD_FRAME_FROM_IMAGE(image_file):
  image = cv2.imread(image_file)
  if image is None: return None
  return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def LOAD_FRAME_FROM_VIDEO(video_file, frame_number_start, num_frames):
  video = cv2.VideoCapture(video_file)
  video.set(cv2.CAP_PROP_POS_FRAMES,frame_number_start)
  frames = list()
  count = 0
  while count < num_frames:
    ret, frame = video.read()
    if ret:
      frames.append(frame)
      count += 1
    else:
      break
  video.release()
  return frames

def LOSS(y1,y2):
  grid = 3
  y1 = y1.numpy()
  y2 = y2.numpy()
  for x in range(y1.shape[0]):
    pred_result = y1[x]
    print(y2[x].numpy())
    target = FLAGS['targets'][y2[x]]
    for i in range(grid):
      conv, pred = pred_result[i*2], pred_result[i*2+1]
      loss_items = utils.COMPUTE_LOSS(pred, conv, *target[i], FLAGS['strides'], FLAGS['iou_loss_threshold'], i)
      giou_loss += loss_items[0]
      conf_loss += loss_items[1]
      prob_loss += loss_items[2]

def TRAIN_ON_BUFFER(model):
  LOAD_BUFFER()
  images = []
  batch_image = list()
  batch_label = list()
  batch_label_sbbox = list()
  batch_label_mbbox = list()
  batch_label_lbbox = list()
  batch_sbboxes = list()
  batch_mbboxes = list()
  batch_lbboxes = list()
  batch_y = list()
  batch_x = list()
  for x,y in FLAGS['buffer']:
    image, bboxes = utils.IMAGE_PREPROCESS(x, (FLAGS['input_size'],FLAGS['input_size']), np.array(list(y)))
    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = utils.PREPROCESS_TRUE_BBOXES(bboxes,FLAGS['input_size'],FLAGS['strides'],FLAGS['anchor_per_scale'],FLAGS['num_classes'],FLAGS['max_bbox_per_scale'],FLAGS['anchors'])
    batch_x.append(image)
    batch_label_sbbox.append(label_sbbox)
    batch_label_mbbox.append(label_mbbox)
    batch_label_lbbox.append(label_lbbox)
    batch_sbboxes.append(sbboxes)
    batch_mbboxes.append(mbboxes)
    batch_lbboxes.append(lbboxes)
  batch_smaller_target = (np.array(batch_label_sbbox),np.array(batch_sbboxes))
  batch_medium_target = (np.array(batch_label_mbbox),np.array(batch_mbboxes))
  batch_larger_target = (np.array(batch_label_lbbox),np.array(batch_lbboxes))
  targets = (batch_smaller_target,batch_medium_target,batch_larger_target)
  FLAGS['targets'] = targets
  batch_x = np.array(batch_x)
  batch_y = np.array([i for i in range(batch_x.shape[0])]).reshape(batch_x.shape[0],1)
  model.fit(x=batch_x,y=batch_y,
            batch_size=FLAGS['batch_size'],
            epochs=FLAGS['num_epochs'],
            validation_split=FLAGS['validation_split'],
            shuffle=True,
            workers=FLAGS['num_workers'],
            use_multiprocessing=FLAGS['multiprocessing'],
            )

def INITIALIZE():
  LOAD_FLAGS()
  DISCOVER_DATASET()

def TRAIN(model):
  model.compile(optimizer='adam',loss=LOSS)
  while DATASET_HAS_MORE():
    TRAIN_ON_BUFFER(model)
    return
