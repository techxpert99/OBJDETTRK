import time
from sys import stdout

class Evaluator:
    def __init__(self):
        pass
    
    def INIT(self, ground_truths_file, dataset_type, detect_file, track_file, evaluation_result_files = None):
        self.EVAL_FLAGS = {'gt_f':ground_truths_file, 'dtype': dataset_type.lower(), 'det_f':detect_file,'tr_f':track_file,'ev_f':evaluation_result_files}
        self.READ_GT()
        self.READ_DET()
        self.READ_TRK()

    def READ_GT(self):
        dtyp = self.EVAL_FLAGS['dtype']
        gtf = self.EVAL_FLAGS['gt_f']

        def mot16tococo17class(cid):
            if cid in [1,2,7]: return 'person'
            elif cid == 4: return 'bicycle'
            elif cid == 3: return 'car'
            elif cid == 5: return 'motorbike'
            return 'unknown'

        if dtyp == 'mot-16':
            gt = dict()
            with open(gtf,'r') as f:
                for line in f.read().splitlines():
                    frn, tid, xmin, ymin, wid, hgt, con, cid =  tuple([int(_) for _ in line.split(',')[:-1]])
                    if frn not in gt:
                        gt[frn] = dict()
                    if tid not in gt[frn]:
                        gt[frn][tid] = set()
                    if con:
                        gt[frn][tid].add((mot16tococo17class(cid),xmin,ymin,xmin+wid,ymin+hgt))
            self.EVAL_FLAGS['gt'] = gt
    
    def READ_DET(self):
        detf = self.EVAL_FLAGS['det_f']
        det = dict()
        with open(detf,'r') as f:
            for line in f.read().splitlines():
                tmp = line.split(',')
                frn,cln = int(tmp[0]),tmp[1]
                xmin,ymin,xmax,ymax = tuple([int(_) for _ in tmp[2:]])
                if frn not in det:
                    det[frn] = set()
                det[frn].add((cln,xmin,ymin,xmax,ymax,))
        self.EVAL_FLAGS['det'] = det
    
    def READ_TRK(self):
        trf = self.EVAL_FLAGS['tr_f']
        tr = dict()
        with open(trf,'r') as f:
            for line in f.read().splitlines():
                tmp = line.split(',')
                frn,id,cln = int(tmp[0]),int(tmp[1]),tmp[2]
                xmin,ymin,xmax,ymax = tuple([int(_) for _ in tmp[3:]])
                if frn not in tr:
                    tr[frn] = dict()
                if id not in tr[frn]:
                    tr[frn][id] = set()
                tr[frn][id].add((cln,xmin,ymin,xmax,ymax))
    
    def IOU(self):
        bb1 = self.EVAL_FLAGS['bb1']
        bb2 = self.EVAL_FLAGS['bb2']
        xmin1,ymin1,xmax1,ymax1 = bb1
        xmin2,ymin2,xmax2,ymax2 = bb2
        xmin,ymin,xmax,ymax = max(xmin1,xmin2),max(ymin1,ymin2),min(xmax1,xmax2),min(ymax1,ymax2)
        inter_w,inter_h = xmax-xmin,ymax-ymin
        w1,h1,w2,h2 = xmax1-xmin1,ymax1-ymin1,xmax2-xmin2,ymax2-ymin2   
        if(inter_w<0 or inter_h<0):
            inter_area = 0
        else:
            inter_area = inter_w*inter_h
        if(w1<0 or h1<0):
            area1 = 0
        else:
            area1 = w1*h1
        if(w1<0 or w2<0):
            area2 = 0
        else:
            area2 = w2*h2
        union_area = max(0,area1+area2-inter_area)

        if union_area == 0:
            self.EVAL_FLAGS['iou_res'] = 1
        else:
            self.EVAL_FLAGS['iou_res'] = inter_area/union_area

    def SELECT_MAX_IOU_BBOX(self):
        bbox_compare = self.EVAL_FLAGS['bb_cmp']
        bbox_list = self.EVAL_FLAGS['bb_lst']
        self.EVAL_FLAGS['bb1'] = bbox_compare
        max_iou_bbox = None
        max_iou = 0
        for bbox in bbox_list:
            self.EVAL_FLAGS['bb2'] = bbox
            self.IOU()
            if self.EVAL_FLAGS['iou_res'] > max_iou:
                max_iou = self.EVAL_FLAGS['iou_res']
                max_iou_bbox = bbox
        self.EVAL_FLAGS['max_iou'] = max_iou
        self.EVAL_FLAGS['max_iou_bbox'] = max_iou_bbox

    def CROSS_MATCH_DETECTIONS(self):
        x1 = self.EVAL_FLAGS['x1']
        x2 = self.EVAL_FLAGS['x2']
        gt_detected_set = set()
        cross_map = list()
        false_positive_set = set()
        for y1 in x2:
            max_iou = 0
            max_bbox = None
            for y2 in x1:
                if y2 in gt_detected_set or y1[0] != y2[0]:
                    continue
                self.EVAL_FLAGS['bb1'] = y1[1:]
                self.EVAL_FLAGS['bb2'] = y2[1:]
                self.IOU()
                if max_iou < self.EVAL_FLAGS['iou_res']:
                    max_iou = self.EVAL_FLAGS['iou_res']
                    max_bbox = y2
            if max_bbox is not None and max_iou >= 0.5:
                cross_map.append((y1,max_bbox,max_iou))
                gt_detected_set.add(max_bbox)
            else:
                false_positive_set.add(y1)
        undetected_gt_set = set(x1)-gt_detected_set
        self.EVAL_FLAGS['cross_res'] = (undetected_gt_set,false_positive_set,cross_map)
    
    def EVAULATE_DETECTIONS_ON_FRAME(self):
        frn = self.EVAL_FLAGS['fr_n']
        gt = self.EVAL_FLAGS['gt']
        det = self.EVAL_FLAGS['det']
        if frn not in gt:
            if frn not in det or not det[frn]:
              self.EVAL_FLAGS['mean_det_iou'] = 1
              self.EVAL_FLAGS['mean_gt_iou'] = 1
            else:
              self.EVAL_FLAGS['mean_det_iou'] = 0
              self.EVAL_FLAGS['mean_gt_iou'] = 1
            return
        elif frn not in det:
            if gt[frn]:
              self.EVAL_FLAGS['mean_det_iou'] = 1
              self.EVAL_FLAGS['mean_gt_iou'] = 0
            else:
              self.EVAL_FLAGS['mean_det_iou'] = 1
              self.EVAL_FLAGS['mean_gt_iou'] = 1
            return
        gtf = gt[frn]
        dtf = det[frn]
        x1 = set()
        for bboxes in gtf.values():
          for bbox in bboxes:
              x1.add(bbox)
        x2 = set()
        for bbox in dtf:
          x2.add(bbox)
        self.EVAL_FLAGS['x1'] = x1
        self.EVAL_FLAGS['x2'] = x2
        self.CROSS_MATCH_DETECTIONS()
        undet_gt,det_fp,cross_map = self.EVAL_FLAGS['cross_res']
        if 'ev_fil' in  self.EVAL_FLAGS and  self.EVAL_FLAGS['ev_fil'] is not None:
          ev_fil = self.EVAL_FLAGS['ev_fil']
          for u in undet_gt:
            u = u[1:]
            ev_fil[0].write(str(frn)+',undet,'+str(u[0])+','+str(u[1])+','+str(u[2])+','+str(u[3])+'\n')
          for u in det_fp:
            u = u[1:]
            ev_fil[1].write(str(frn)+',fpos,'+str(u[0])+','+str(u[1])+','+str(u[2])+','+str(u[3])+'\n')
          i=1
          for u,v,w in cross_map:
            u = u[1:]
            v = v[1:]
            ev_fil[2].write(str(frn)+f',pr-{i},'+str(u[0])+','+str(u[1])+','+str(u[2])+','+str(u[3])+'\n')
            ev_fil[3].write(str(frn)+f',gt-{i},'+str(v[0])+','+str(v[1])+','+str(v[2])+','+str(v[3])+'\n')
            i += 1
        avg_iou = 0
        for u,v,w in cross_map:
          avg_iou += w
        avg_iou /= len(cross_map)

        num_gt = len(x1)
        num_det = len(x2)
        num_fn = len(undet_gt)
        num_fp = len(det_fp)

        self.EVAL_FLAGS['mean_iou'] = avg_iou
        self.EVAL_FLAGS['num_fp'] = num_fp
        self.EVAL_FLAGS['num_fn'] = num_fn
        self.EVAL_FLAGS['num_gt'] = num_gt
        self.EVAL_FLAGS['num_det'] = num_det

    def SET_FLAG(self, flag, value):
        self.EVAL_FLAGS[flag] = value
    
    def GET_FLAG(self, flag):
        if flag in self.EVAL_FLAGS: return self.EVAL_FLAGS[flag]
        return None
    
    def RESET_FLAGS(self):
        self.EVAL_FLAGS = dict()

    def EVALUATE_DETECTIONS(self):
        num_frames = max(len(self.EVAL_FLAGS['gt']),len(self.EVAL_FLAGS['det']))
        if self.EVAL_FLAGS['ev_f'] is not None and len(self.EVAL_FLAGS['ev_f'])>=4:
            self.EVAL_FLAGS['ev_fil'] = [open(self.EVAL_FLAGS['ev_f'][0],'w'),open(self.EVAL_FLAGS['ev_f'][1],'w'),open(self.EVAL_FLAGS['ev_f'][2],'w'),open(self.EVAL_FLAGS['ev_f'][3],'w')]
        print(f'Evaluating Detections on {num_frames} frames:')
        start_time = time.time()
        
        mean_iou = 0
        num_fp = 0
        num_gt = 0
        num_det = 0
        num_fn = 0
        for frame_number in range(1,num_frames+1):
            self.EVAL_FLAGS['fr_n'] = frame_number
            self.EVAULATE_DETECTIONS_ON_FRAME()
            mean_iou += self.EVAL_FLAGS['mean_iou']
            num_fp += self.EVAL_FLAGS['num_fp']
            num_gt += self.EVAL_FLAGS['num_gt']
            num_det += self.EVAL_FLAGS['num_det']
            num_fn += self.EVAL_FLAGS['num_fn']
            end_time = time.time()
            stdout.write(f'\rFrames evaluated: {frame_number}, Rate: {frame_number/(end_time-start_time): .2f} fps')
        
        mean_iou /= num_frames
        mean_fp = num_fp/num_det
        mean_fn = num_fn/num_gt
        AP = mean_iou * (num_det-num_fp)/(mean_iou * (num_det-num_fp)+num_fp)

        if self.EVAL_FLAGS['ev_f'] is not None and len(self.EVAL_FLAGS['ev_f'])>=4:
          for f in self.EVAL_FLAGS['ev_fil']:
            f.close()

        print('\rFinished evaluating detections: ')
        print(f'Total Frames: {num_frames}')
        print(f'Frame Rate: {num_frames/(end_time-start_time): .2f} fps')
        print(f'Average Precision: {AP:.2f}')
        print(f'Mean Detection IOU: {mean_iou:.2f}')
        print(f'Fraction of Ground Truths Undetected: {mean_fn:.2f}')
        print(f'Fraction of Detections Unmatched: {mean_fp:.2f}')

