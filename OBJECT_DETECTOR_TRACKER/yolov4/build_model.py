import tensorflow as tf
from yolov4 import YOLO, DECODE
from utils import FILTER_BOXES
import utils
import trainer

FLAGS = dict()

def LOAD_FLAGS():
  global FLAGS
  config = utils.READ_CONFIG_FILE(utils.CONFIG_FILENAME)
  if 'MODEL.PRE_TRAINED_WEIGHTS' in config:
    FLAGS['weights'] = config['MODEL.PRE_TRAINED_WEIGHTS']
  if 'MODEL.TRAIN' in config:
    FLAGS['trainable'] = config['MODEL.TRAIN'] == 'True'
  FLAGS['output'] = config['MODEL.OUTPUT']
  FLAGS['input_size'] = utils.PARSE_ITEM(config['MODEL.INPUT_SIZE'],int)
  FLAGS['score_threshold'] = utils.PARSE_ITEM(config['MODEL.SCORE_THRESHOLD'],float)

def BUILD_MODEL():
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.LOAD_CONFIG()
    LOAD_FLAGS()
    input_layer = tf.keras.layers.Input([FLAGS['input_size'], FLAGS['input_size'], 3])
    feature_maps = YOLO(input_layer, NUM_CLASS)
    bbox_tensors = []
    prob_tensors = []
    output_sizes = FLAGS['input_size'] // STRIDES
    for i, fm in enumerate(feature_maps):
        output_tensors = DECODE(fm, output_sizes[i], NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    boxes, pred_conf = FILTER_BOXES(pred_bbox, pred_prob, score_threshold=FLAGS['score_threshold'], input_shape=tf.constant([FLAGS['input_size'], FLAGS['input_size']]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    TRAIN_MODEL(model)
    model.save(FLAGS['output'])

def LOAD_PRETRAINED(model):
  if 'weights' in FLAGS:
    utils.LOAD_WEIGHTS(model, FLAGS['weights'])

def TRAIN_MODEL(model):
  LOAD_PRETRAINED(model)
  if not 'trainable' in FLAGS: return
  if not FLAGS['trainable']: return
  utils.UNFREEZE_ALL(model)
  trainer.INITIALIZE()
  trainer.TRAIN(model)

def main():
    BUILD_MODEL()

if __name__ == '__main__':
    main()
