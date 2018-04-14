#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

# CLASSES = ('__background__',  # always index 0
#                      'dog', 'person', 'cat', 
#                      'tv', 'car', 'meatballs', 
#                      'marinara sauce', 'tomato soup', 'chicken noodle soup',
#                      'french onion soup', 'chicken breast', 'ribs', 
#                      'pulled pork', 'hamburger', 'cavity')

CLASSES = ('__background__', 'car')

NETS = {
    'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
    'res101': ('res101_faster_rcnn_iter_10000.ckpt',),}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),
'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
'car_track1': ('car_track1_train',),
'tiny_car_track1':('tiny_car_track1_train',),
'horizontal_car_track1': ('horizontal_car_track1_train',)}

def vis_detections(im, class_name, dets, info, idx, thresh=0.5):

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
      return
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in inds:
      bbox = dets[i, :4]
      score = dets[i, -1]

      info.append(np.append(dets[i], idx))

      cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
      # cv2.putText(im, class_name, (bbox[0], bbox[1]), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
      cls_ind += 1 # because we skipped background
      cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
      cls_scores = scores[:, cls_ind]
      dets = np.hstack((cls_boxes,
                        cls_scores[:, np.newaxis])).astype(np.float32)
      keep = nms(dets, NMS_THRESH)
      dets = dets[keep, :]
      vis_detections(im, cls, dets, thresh=CONF_THRESH)

def my_demo(sess, net, im, video_name, idx, info):
  scores, boxes = im_detect(sess, net, im)
  CONF_THRESH = 0.8
  NMS_THRESH = 0.3
  for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    vis_detections(im, cls, dets, info, idx, thresh=CONF_THRESH)

    # cv2.imwrite(os.path.join(video_name, str(idx).zfill(5) + '.jpeg'), im)    

    # vis_detections(im, cls, dets, thresh=CONF_THRESH)    
    # cv2.imshow('img', vis_detections(im, cls, dets, thresh=CONF_THRESH))
    # cv2.waitKey(0)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", len(CLASSES),
                          tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # info = []


    # image_path = '/media/ad/DATA/aicitychallenge/tf-faster-rcnn/Loc3_1/00010.jpeg'
    # image = cv2.imread(image_path)
    # image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))
    # my_demo(sess, net, image, 'Loc3_1', 999999, info)

    # exit(0)
    video_names = os.listdir('/media/ad/DATA/aicitychallenge/tf-faster-rcnn/data/track1_videos')
    for video_name in video_names:
      if (video_name[:4] != 'Loc3'):
        continue
      if not os.path.exists(video_name[:-4]):
        os.mkdir(video_name[:-4])

      video_path = os.path.join('/media/ad/DATA/aicitychallenge/tf-faster-rcnn/data/track1_videos', video_name)
      print(video_path)

      info = []
      video = cv2.VideoCapture(video_path)
      idx = 1
      while (video.isOpened()):
        isReadable, frame = video.read()
        if not isReadable:
          break
        frame = frame[: 540, 480: 1440,:]
        sz = frame.shape
        frame = cv2.resize(frame, (sz[1]* 3, sz[0] * 3))
        my_demo(sess, net, frame, video_name[:-4], idx, info)
        # frame = cv2.resize(frame, (sz[1], sz[0]))
        idx += 1

      np.save(os.path.join(video_name[:-4], 'info_' + video_name[:-4]), np.array(info))
    
