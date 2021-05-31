import cv2
from matplotlib.pyplot import get_cmap
import numpy as np

class Visualizer:
    # Visualizer Flags:
    #   (1) input_image_filename *
    #   (2) input_image
    #   (3) input_track_filename *
    #   (4) input_tracks
    #   (5) input_object_filename
    #   (6) input_objects
    #   (7) frame_number
    #   (8) color_list *
    #   (9) detection_color *
    #   (10) output_image_filename *
    #   (11) output_image
    #   (12) video_codec

    def __init__(self):
        pass

    def INIT(self, input_object_file, input_track_file):
        self.RESET_FLAGS()
        self.SET_DEFAULT_COLOR_LIST()
        self.SET_FLAG('detection_color',(255,0,0))
        self.SET_FLAG('input_object_filename',input_object_file)
        self.SET_FLAG('input_track_filename',input_track_file)
        self.SET_FLAG('video_codec','XVID')
        if input_object_file is not None:
          self.READ_OBJECTS()
        if input_track_file is not None:
          self.READ_TRACKS()
    
    def SET_FLAG(self,flag,value):
        self.VISUALIZER_FLAGS[flag] = value
    
    def GET_FLAG(self,flag):
        if flag in self.VISUALIZER_FLAGS: return self.VISUALIZER_FLAGS[flag]
        return None
    
    def RESET_FLAGS(self):
        self.VISUALIZER_FLAGS = dict()
    
    def READ_IMAGE(self):
        self.SET_FLAG('input_image',cv2.cvtColor(cv2.imread(self.GET_FLAG('input_image_filename')),cv2.COLOR_BGR2RGB))
    
    def WRITE_IMAGE(self):
        cv2.imwrite(self.GET_FLAG('output_image_filename'), cv2.cvtColor(self.GET_FLAG('output_image').copy(), cv2.COLOR_RGB2BGR))

    def READ_TRACKS(self):
        with open(self.GET_FLAG('input_track_filename')) as trackfile:
            tracks = dict()
            for track in trackfile.read().splitlines():
                tmp = track.split(',')
                frame_num,track_id = tuple([int(_) for _ in tmp[:2]])
                class_name = tmp[2]
                bbox = tuple([int(_) for _ in tmp[3:]])
                if frame_num not in tracks:
                    tracks[frame_num] = list()
                tracks[frame_num].append({'track_id':track_id,'class_name':class_name,'bbox':bbox})
            self.SET_FLAG('input_tracks',tracks)

    def READ_OBJECTS(self):
        with open(self.GET_FLAG('input_object_filename')) as objectfile:
            detects = dict()
            for object in objectfile.read().splitlines():
                tmp = object.split(',')
                frame_num = int(tmp[0])
                class_name = tmp[1]
                bbox = tuple([int(_) for _ in tmp[2:]])
                if frame_num not in detects:
                    detects[frame_num] = list()
                detects[frame_num].append({'class_name':class_name,'bbox':bbox})
            self.SET_FLAG('input_objects',detects)

    def VISUALIZE_TRACKS(self):
        if self.GET_FLAG('input_tracks') is None: return
        img = self.GET_FLAG('input_image').copy()
        frame_num = self.GET_FLAG('frame_number')
        tracks = self.GET_FLAG('input_tracks')
        color_list = self.GET_FLAG('color_list')
        self.SET_FLAG('output_image',img)
        if frame_num not in tracks:
          return
        for track in tracks[frame_num]:
            id = track['track_id']
            cl = track['class_name']
            n = len(color_list)
            b = track['bbox']
            color = color_list[id%n]
            cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),color,2)
            cv2.rectangle(img,(b[0],b[1]-30),(b[0]+(len(cl)+len(str(id)))*17,b[1]),color,-1)
            cv2.putText(img,cl+"-"+str(id),(b[0],b[1]-10),0,0.75,(255,255,255),2)
    
    def VISUALIZE_OBJECTS(self):
        if self.GET_FLAG('input_objects') is None: return
        img = self.GET_FLAG('input_image').copy()
        frame_num = self.GET_FLAG('frame_number')
        objects = self.GET_FLAG('input_objects')
        color = self.GET_FLAG('detection_color')
        self.SET_FLAG('output_image',img)
        if frame_num not in objects:
          return
        for object in objects[frame_num]:
            cl = object['class_name']
            b = object['bbox']
            cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),color,2)
            cv2.rectangle(img,(b[0],b[1]-30),(b[0]+len(cl)*17,b[1]),color,-1)
            cv2.putText(img,cl,(b[0],b[1]-10),0,0.75,(255,255,255),2)

    def SET_DEFAULT_COLOR_LIST(self):
        cmap = get_cmap('tab20b')
        col_list = [cmap(i)[:3] for i in np.linspace(0,1,20)]
        col_list = [(int(r*255),int(g*255),int(b*255)) for r,g,b in col_list]
        self.SET_FLAG('color_list',col_list)
    
    def RUN_ON_IMAGE(self, input_image_file, output_image_detect_file, output_image_track_file, frame_number=1):
        self.SET_FLAG('input_image_filename',input_image_file)
        self.SET_FLAG('output_image_filename',output_image_detect_file)
        self.SET_FLAG('frame_number',frame_number)
        self.READ_IMAGE()
        if output_image_detect_file is not None:
          self.VISUALIZE_OBJECTS()
          self.WRITE_IMAGE()
        if output_image_track_file is not None:
          self.VISUALIZE_TRACKS()
          self.SET_FLAG('output_image_filename',output_image_track_file)
          self.WRITE_IMAGE()
    
    def RUN_ON_VIDEO(self, input_video_file, output_video_detect_file, output_video_track_file):
        input_video = cv2.VideoCapture(input_video_file)
        width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*self.GET_FLAG('video_codec'))
        output_detect_video = cv2.VideoWriter(output_video_detect_file, codec, fps, (width, height))
        output_track_video = cv2.VideoWriter(output_video_track_file, codec, fps, (width, height))
        frame_num = 0
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret: break
            frame_num += 1
            self.SET_FLAG('input_image',cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            self.SET_FLAG('frame_number',frame_num)
            self.VISUALIZE_OBJECTS()
            output_detect_video.write(self.GET_FLAG('output_image'))
            self.VISUALIZE_TRACKS()
            output_track_video.write(self.GET_FLAG('output_image'))
        input_video.release()
        output_detect_video.release()
        output_track_video.release()
      
    def RUN_ON_IMAGE_SEQUENCE(self, input_image_sequence, output_image_sequence_detect_files, output_image_sequence_track_files,frame_numbers = None):
      for i in range(len(input_image_sequence)):
        if frame_numbers is not None:
          frame_num = frame_numbers[i]
        else:
          frame_num = i+1
        if self.GET_FLAG('input_objects') is not None and self.GET_FLAG('input_tracks') is not None:
          self.RUN_ON_IMAGE(input_image_sequence[i],output_image_sequence_detect_files[i],output_image_sequence_track_files[i],frame_num)
        elif self.GET_FLAG('input_tracks') is None:
          self.RUN_ON_IMAGE(input_image_sequence[i],output_image_sequence_detect_files[i],None,frame_num)
        elif self.GET_FLAG('input_objects') is None:
          self.RUN_ON_IMAGE(input_image_sequence[i],None,output_image_sequence_track_files[i],frame_num)
