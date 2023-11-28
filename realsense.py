from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *

from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
from CentroidTracker import CentroidTracker

#text_color = (0, 255, 0)
thickness = 3

classes = load_classes('data/coco.names')
class_count = len(classes)

def text_color(s):
    val = hash(s) % 0xFFFFFF
    return ((val & 0xFF0000) >> 16, (val & 0xFF00) >> 8, (val & 0xFF))

def get_text_color(dic, text):    
    color = dic.get(text)
    if color is None:
        color = text_color(text)
        dic[text] = color
    return color

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(c1, c2, cls, img, classes, color_dic):
    label = "{0}".format(classes) if cls < class_count else "Unkwon"
    '''
    color = (0, 255, 0)
    cv2.rectangle(img, c1, c2, color, thickness)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.5 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, "person", (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1.5, [225,255,255], 2)
    # check person
    if label in ['person']:
        color = (0, 255, 0)
        cv2.rectangle(img, c1, c2, color, thickness)
        t_size = cv2.getTextSize("person", cv2.FONT_HERSHEY_PLAIN, 1.5 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1.5, [225,255,255], 2)
        return img
    # check another
    else:
        color = get_text_color(color_dic, label)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img
    '''
    return img

def yolo_output(frame, model, confidence, nms_thesh, CUDA, inp_dim, is_draw):
    """
    Get the labeled image and the bounding box coordinates.
    """
    num_classes = 80
    bbox_attrs = 5 + num_classes
    img, orig_im, dim = prep_image(frame, inp_dim)

    im_dim = torch.FloatTensor(dim).repeat(1,2)

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim)) / inp_dim

    output[:,[1,3]] *= frame.shape[1]
    output[:,[2,4]] *= frame.shape[0]

    boxs = list([])
    color_dic = {}
            
    for i in range(output.shape[0]):
        c1 = tuple(output[i,1:3].int())
        c2 = tuple(output[i,3:5].int())
        box = [c1[0].item(), c1[1].item(), c2[0].item(), c2[1].item()]
        
        # If the starting point of the coordinate is 0,0, it is ignored.
        if not any([box[0], box[2]]) or c1[0].item() == c2[0].item() or c1[1].item() == c2[1].item():
            continue
        
        if is_draw:
            write(c1, c2, output[i, -1], orig_im, num_classes, color_dic)
        #print(output[i, -1])
        if int(output[i, -1]) == 0:
            boxs.append(box)

    return orig_im, boxs

