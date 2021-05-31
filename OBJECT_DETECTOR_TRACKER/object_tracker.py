import os
import sys
sys.path.append(os.getcwd()+'/yolov4')
os.chdir(os.getcwd()+'/yolov4')

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import utils as utils
from utils import FILTER_BOXES

from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import generate_detections as gdet

from sys import stdout

cfg = utils.READ_CONFIG_FILE(utils.CONFIG_FILENAME)

def class_eq(gt_class_no, pr_class_name):
    if pr_class_name=='bicycle' and gt_class_no in {4,6}:
        return True
    if pr_class_name=='car' and gt_class_no==3:
        return True
    if pr_class_name=='motorbike' and gt_class_no==5:
        return True
    if pr_class_name=='skateboard' and gt_class_no==6:
        return True
    return False

# The Object Detector and Tracker
class ObjectTracker:

    # Global Object Tracker Flags:
    #   (1) tracker
    #   (2) infer
    #   (3) input_size
    #   (4) iou_threshold
    #   (5) score_threshold
    #   (6) class_names
    #   (7) allowed_classes
    #   (8) input
    #   (9) output_track_filename
    #   (10) frame_number
    #   (11) encoder
    #   (12) nms_max_overlap
    #   (13) detections
    #   (14) output_object_filename
    #   (15) detections
    
    def __init__(self):
        self.OBJ_TRACKER_FLAGS = dict()

    # Initializes the Object Detector/ Tracker
    def INIT_OBJECT_TRACKER(self, weights_path='./save/yolov4-416',model_filename='../deep_sort/mars-small128.pb'):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        
        # Initialize deep sort
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        
        # Calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        # Initialize tracker
        tracker = Tracker(metric)

        # Load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.LOAD_CONFIG()

        # Load tf saved model
        saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        infer._backref_to_saved_model = saved_model_loaded

        # Read in all class names from config
        class_names = utils.READ_CLASS_NAMES(cfg['YOLO.CLASSES'])

        # Initialize flags
        self.RESET_FLAGS()

        # Set loaded flags
        self.OBJ_TRACKER_FLAGS['infer'] = infer
        self.OBJ_TRACKER_FLAGS['tracker'] = tracker
        self.OBJ_TRACKER_FLAGS['class_names'] = class_names
        self.OBJ_TRACKER_FLAGS['allowed_classes'] = class_names.values()
        self.OBJ_TRACKER_FLAGS['encoder'] = encoder
        self.OBJ_TRACKER_FLAGS['nms_max_overlap'] = nms_max_overlap

    # Writes the tracked object to the output file
    # Format: frame number, track id, class name, xmin, ymin, xmax, ymax
    def WRITE_TRACK(self,bbox,classname,id):
        str_bbox = str(int(bbox[0]))+','+str(int(bbox[1]))+','+str(int(bbox[2]))+','+str(int(bbox[3]))
        str_classname=str(classname)
        str_id = str(id)
        str_framenumber = str(self.OBJ_TRACKER_FLAGS['frame_number'])
        self.OBJ_TRACKER_FLAGS['output_track_file'].write(str_framenumber+','+str_id+','+str_classname+','+str_bbox+'\n')

    # Writes the detected object to the output file
    # Format: frame number, class name, xmin, ymin, xmax, ymax
    def WRITE_OBJECT(self,bbox,classname):
        str_bbox = str(int(bbox[0]))+','+str(int(bbox[1]))+','+str(int(bbox[0]+bbox[2]))+','+str(int(bbox[1]+bbox[3]))
        str_classname=str(classname)
        str_framenumber = str(self.OBJ_TRACKER_FLAGS['frame_number'])
        self.OBJ_TRACKER_FLAGS['output_object_file'].write(str_framenumber+','+str_classname+','+str_bbox+'\n')

    # Used to set an object tracker flag
    def SET_FLAG(self,flag, value):
        self.OBJ_TRACKER_FLAGS[flag] = value

    # Returns the value of an object tracker flag. Returns none if no such flag currently exists
    def GET_FLAG(self,flag):
        if flag in self.OBJ_TRACKER_FLAGS:
            return self.OBJ_TRACKER_FLAGS[flag]
        return None

    # Resets all flags
    def RESET_FLAGS(self):
        self.OBJ_TRACKER_FLAGS['input_size'] = 416
        self.OBJ_TRACKER_FLAGS['iou_threshold'] = 0.45
        self.OBJ_TRACKER_FLAGS['score_threshold'] = 0.5
        self.OBJ_TRACKER_FLAGS['frame_number'] = 1

        self.OBJ_TRACKER_FLAGS['input'] = None
        self.OBJ_TRACKER_FLAGS['output_track_file'] = None
        self.OBJ_TRACKER_FLAGS['output_object_file'] = None
        self.OBJ_TRACKER_FLAGS['infer'] = None
        self.OBJ_TRACKER_FLAGS['tracker'] = None
        self.OBJ_TRACKER_FLAGS['class_names'] = None
        self.OBJ_TRACKER_FLAGS['allowed_classes'] = None
        self.OBJ_TRACKER_FLAGS['encoder'] = None
        self.OBJ_TRACKER_FLAGS['nms_max_overlap'] = None
        self.OBJ_TRACKER_FLAGS['detections'] = None

    # Detects and tracks object in a single frame/ image [Controlled by setting OBJ_TRACKER_FLAGS]
    def DETECT_TRACK_OBJECT(self):
        # Load required flags
        input_size = self.OBJ_TRACKER_FLAGS['input_size']
        infer = self.OBJ_TRACKER_FLAGS['infer']
        iou_threshold = self.OBJ_TRACKER_FLAGS['iou_threshold']
        score_threshold = self.OBJ_TRACKER_FLAGS['score_threshold']
        class_names = self.OBJ_TRACKER_FLAGS['class_names']
        allowed_classes = self.OBJ_TRACKER_FLAGS['allowed_classes']
        input_data = self.OBJ_TRACKER_FLAGS['input']
        encoder = self.OBJ_TRACKER_FLAGS['encoder']
        nms_max_overlap = self.OBJ_TRACKER_FLAGS['nms_max_overlap']

        # Convert image from BGR to RGB
        frame = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

        # Resize and normalize image
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # Predict the output on the image
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        # Process the output
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold
            )

            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # Format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.FORMAT_BOXES(bboxes, original_h, original_w)

            # Store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # Loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)

            # Delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # Writing the detected objects to the output file
            for i in range(count):
              self.WRITE_OBJECT(bboxes[i],names[i])
            
            # Encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            # Run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            self.OBJ_TRACKER_FLAGS['detections'] = detections
            self.TRACK_OBJECT()
            self.OBJ_TRACKER_FLAGS['detections'] = None

    def TRACK_OBJECT(self):
        tracker = self.OBJ_TRACKER_FLAGS['tracker']
        detections = self.OBJ_TRACKER_FLAGS['detections']
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_bbox = track.to_tlbr()
            track_classname = track.get_class()
            track_id = track.track_id
            self.WRITE_TRACK(track_bbox,track_classname,track_id)

    # Runs the pre-initialized tracker on an image
    def RUN_ON_IMAGE(self, input_filename, output_object_filename, output_track_filename):
        print(f'Running Object Tracker on: "{input_filename}"')
        self.OBJ_TRACKER_FLAGS['input'] = cv2.imread(input_filename)
        start_time = time.time()
        processing_time_elapsed = 0
        self.OBJ_TRACKER_FLAGS['frame_number'] = 1
        with open(output_track_filename,'w') as output_track_file, open(output_object_filename,'w') as output_object_file:
            self.OBJ_TRACKER_FLAGS['output_track_file'] = output_track_file
            self.OBJ_TRACKER_FLAGS['output_object_file'] = output_object_file
            processing_time_elapsed = time.time()
            self.DETECT_TRACK_OBJECT()
            processing_time_elapsed = time.time() - processing_time_elapsed
            self.OBJ_TRACKER_FLAGS['output_track_file'] = None
            self.OBJ_TRACKER_FLAGS['output_object_file'] = None
        self.OBJ_TRACKER_FLAGS['input'] = None
        self.OBJ_TRACKER_FLAGS['frame_number'] = None
        
        end_time = time.time()
        print(f'Run finished. Outputs written to: "{output_object_filename}", "{output_track_filename}" respectively')
        print(f'Time Elapsed: {end_time-start_time:.2f} s')
        print(f'Detection/ Tracking Frame Rate: {1/processing_time_elapsed:.2f} fps')

    # Runs the pre-initialized tracker on a video
    def RUN_ON_VIDEO(self, input_filename, output_object_filename, output_track_filename):
        print(f'Running Object Tracker on: "{input_filename}"')
        start_time = time.time()
        processing_time_elapsed = 0
        vid = cv2.VideoCapture(input_filename)
        with open(output_track_filename,'w') as output_track_file, open(output_object_filename,'w') as output_object_file:
            frame_num = 0
            self.OBJ_TRACKER_FLAGS['output_track_file'] = output_track_file
            self.OBJ_TRACKER_FLAGS['output_object_file'] = output_object_file
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                  break
                frame_num += 1
                self.OBJ_TRACKER_FLAGS['input'] = frame
                self.OBJ_TRACKER_FLAGS['frame_number'] = frame_num
                tmp_time_elapsed = time.time()
                self.DETECT_TRACK_OBJECT()
                processing_time_elapsed += (time.time()-tmp_time_elapsed)/2
                self.OBJ_TRACKER_FLAGS['input'] = None
                self.OBJ_TRACKER_FLAGS['frame_number'] = None
                end_time = time.time()
                stdout.write(f'\rFrame #{frame_num}: {frame_num/processing_time_elapsed:.2f} fps'+' '*10)
            self.OBJ_TRACKER_FLAGS['output_track_file'] = None
            self.OBJ_TRACKER_FLAGS['output_object_file'] = None
            vid.release()
        
        end_time = time.time()
        print(f'\rRun finished. Outputs written to: "{output_object_filename}", "{output_track_filename}" respectively')
        print(f'Total Time Elapsed: {end_time-start_time:.2f} s')
        print(f'Total Frames: {frame_num}')
        print(f'Detection/ Tracking Frame Rate: {frame_num/processing_time_elapsed:.2f} fps')

    # Runs the pre-initialized tracker on an image_sequence
    def RUN_ON_IMAGE_SEQUENCE(self, input_image_filenames, output_object_filename, output_track_filename):
        print(f'Running Object Tracker on the given image batch of size: {len(input_image_filenames)}')
        start_time = time.time()
        processing_time_elapsed = 0

        with open(output_track_filename,'w') as output_track_file, open(output_object_filename,'w') as output_object_file:
            frame_num = 0
            self.OBJ_TRACKER_FLAGS['output_track_file'] = output_track_file
            self.OBJ_TRACKER_FLAGS['output_object_file'] = output_object_file
            for img_f in input_image_filenames:
                frame = cv2.imread(img_f)
                frame_num += 1
                self.OBJ_TRACKER_FLAGS['input'] = frame
                self.OBJ_TRACKER_FLAGS['frame_number'] = frame_num
                tmp_time_elapsed = time.time()
                self.DETECT_TRACK_OBJECT()
                processing_time_elapsed += (time.time()-tmp_time_elapsed)/2
                self.OBJ_TRACKER_FLAGS['input'] = None
                self.OBJ_TRACKER_FLAGS['frame_number'] = None
                stdout.write(f'\rImage #{frame_num}: {frame_num/processing_time_elapsed:.2f} fps'+' '*10)
            self.OBJ_TRACKER_FLAGS['output_track_file'] = None
            self.OBJ_TRACKER_FLAGS['output_object_file'] = None
        end_time = time.time()
        print(f'\rRun finished. Outputs written to: "{output_object_filename}", "{output_track_filename}" respectively')
        print(f'Total Time Elapsed: {end_time-start_time:.2f} s')
        print(f'Total Images: {frame_num}')
        print(f'Detection/ Tracking Frame Rate: {frame_num/processing_time_elapsed:.2f} fps')
