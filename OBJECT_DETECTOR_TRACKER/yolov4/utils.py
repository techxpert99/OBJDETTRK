import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
import re

CONFIG_FILENAME = 'config.txt'

def LOAD_FREEZE_LAYER():
    freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    return freeze_layouts

def LOAD_WEIGHTS(model, weights_file):
    layer_size = 110
    output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'
        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]
        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])
    wf.close()

def READ_CLASS_NAMES(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def READ_CONFIG_FILE(config_filename):
  config = dict()
  ws = f'(?:[\\s\\t\\r\\f]*)'
  key = f'([A-Za-z0-9\\$\\.\\_]*)'
  anych = f'(?:.|[\\s\\t\\r\\f])'
  qval = f'(?:"((?:[^"\\\\]|\\\\{anych})*)")'
  uval = f'([^\\s\\t\\r\\f]*)'
  val = f'(?:{qval}|{uval})'
  com = f'(\\#{anych}*)'
  config_line_syntax = f'(?:{ws}(?:{key}{ws}={ws}{val})?{ws}{com}?)'
  pat = re.compile(config_line_syntax)
  with open(config_filename,'r') as f:
    for line in f.read().splitlines():
      mat = pat.match(line)
      if mat:
        xkey = mat.group(1)
        if mat.group(2): xval = mat.group(2)
        else: xval = mat.group(3)
        if xkey is not None:
          config[xkey] = xval
  return config

def GET_ANCHORS(anchors_path):
    anchors = np.array(anchors_path)
    return anchors.reshape(3, 3, 2)

def PARSE_ITEM(item, dtype_parser):
  return dtype_parser(item.strip())

def PARSE_LIST(string_list, dtype_parser):
  return [PARSE_ITEM(item,dtype_parser) for item in string_list.split(',')]

def LOAD_CONFIG():
    config = READ_CONFIG_FILE(CONFIG_FILENAME)
    config['YOLO.STRIDES'] = PARSE_LIST(config['YOLO.STRIDES'],int)
    config['YOLO.ANCHORS'] = PARSE_LIST(config['YOLO.ANCHORS'],int)
    config['YOLO.XYSCALE'] = PARSE_LIST(config['YOLO.XYSCALE'],float)
    STRIDES = np.array(config['YOLO.STRIDES'])
    ANCHORS = GET_ANCHORS(config['YOLO.ANCHORS'])
    XYSCALE = config['YOLO.XYSCALE']
    NUM_CLASS = len(READ_CLASS_NAMES(config['YOLO.CLASSES']))
    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def IMAGE_PREPROCESS(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.
    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def FORMAT_BOXES(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height
    return bboxes

def BBOX_IOU(bboxes1, bboxes2):
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]
    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = bboxes1_area + bboxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)
    return iou

def BBOX_GIOU(bboxes1, bboxes2):
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]
    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = bboxes1_area + bboxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)
    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )
    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]
    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)
    return giou


def BBOX_CIOU(bboxes1, bboxes2):
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]
    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = bboxes1_area + bboxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)
    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )
    enclose_section = enclose_right_down - enclose_left_up
    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2
    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]
    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2
    diou = iou - tf.math.divide_no_nan(rho_2, c_2)
    v = (
        (
            tf.math.atan(
                tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
            )
            - tf.math.atan(
                tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
            )
        )
        * 2
        / np.pi
    ) ** 2
    alpha = tf.math.divide_no_nan(v, 1 - iou + v)
    ciou = diou - alpha * v
    return ciou

def NMS(bboxes, iou_threshold, sigma=0.3, method='nms'):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []
    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)
            assert method in ['nms', 'soft-nms']
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes

def PREPROCESS_TRUE_BBOXES(bboxes,input_size,strides,anchor_per_scale,num_classes,max_bbox_per_scale,anchors):
  train_output_sizes = input_size//strides
  label = [np.zeros((train_output_sizes[i], train_output_sizes[i], anchor_per_scale,
                      5 + num_classes)) for i in range(3)]
  bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(3)]
  bbox_count = np.zeros((3,))
  for bbox in bboxes:
      bbox_coor = bbox[:4]
      bbox_class_ind = int(bbox[4])
      onehot = np.zeros(num_classes, dtype=np.float)
      onehot[bbox_class_ind] = 1.0
      uniform_distribution = np.full(num_classes, 1.0 / num_classes)
      deta = 0.01
      smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
      bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
      bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
      iou = []
      exist_positive = False
      for i in range(3):
          anchors_xywh = np.zeros((anchor_per_scale, 4))
          anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
          anchors_xywh[:, 2:4] = anchors[i]
          iou_scale = BBOX_IOU(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
          iou.append(iou_scale)
          iou_mask = iou_scale > 0.3
          if np.any(iou_mask):
              xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
              label[i][yind, xind, iou_mask, :] = 0
              label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
              label[i][yind, xind, iou_mask, 4:5] = 1.0
              label[i][yind, xind, iou_mask, 5:] = smooth_onehot
              bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
              bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
              bbox_count[i] += 1
              exist_positive = True
      if not exist_positive:
          best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
          best_detect = int(best_anchor_ind / anchor_per_scale)
          best_anchor = int(best_anchor_ind % anchor_per_scale)
          xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
          label[best_detect][yind, xind, best_anchor, :] = 0
          label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
          label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
          label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
          bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
          bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
          bbox_count[best_detect] += 1
  label_sbbox, label_mbbox, label_lbbox = label
  sbboxes, mbboxes, lbboxes = bboxes_xywh
  return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

def FREEZE_ALL(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            FREEZE_ALL(l, frozen)
def UNFREEZE_ALL(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            UNFREEZE_ALL(l, frozen)

def COMPUTE_LOSS(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]
    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]
    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]
    giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)
    iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
    return giou_loss, conf_loss, prob_loss

def FILTER_BOXES(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)
    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])
    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)
    input_shape = tf.cast(input_shape, dtype=tf.float32)
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    return (boxes, pred_conf)