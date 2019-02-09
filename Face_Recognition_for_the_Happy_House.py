# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:Kun_J
@file:.py
@ide:untitled3
@time:2019-02-05 22:35:00
@month:二月
"""
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format("channels_first")  ###这里使用通道优先
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

# 获取模型

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:",FRmodel.count_params())####获取模型的参数总数


# 1、下面定义triplet_loss
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implemention of the triplet loss as defined by formula
    :param y_true: true labels
    :param y_pred: python list containing three object:
                    anchor: the encodings for the anchor images, of shape(None, 128)
                    positive: the encodings for the positive images, of shape(None, 128)
                    negative: the encodings for the negative images, of shape(None, 128)
    :param alpha:
    :return:
    loss: real numbers, values of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### shart code ###
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.subtract(anchor, positive), axis=-1)
    # Step 2: Computer the (encoding) distance betwween the anchor and the negative
    neg_dist = tf.reduce_sum(tf.subtract(anchor, negative), axis=-1)
    # Step 3: Subtract the two previous distances and add alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### end code ###
    return loss

def test_triblet_loss():
    with tf.Session() as sess:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
                  tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
                  tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
        loss = triplet_loss(y_true, y_pred, alpha=0.2)
        print("loss:" + str(loss.eval()))
        return loss
test_triblet_loss()

# 2、Loading the trained model
"""
FaceNet是通过最小化三元组损失函数来训练的，但是由于训练需要大量的数据和时间，所以我们不会从头开始训练，
相反，我们会加载一个已经训练好的模型，运行下列代码来加载模型，只需要几分钟的时间
"""
import time
start_time = time.clock()

#编译模型
FRmodel.compile(optimizer="adam", loss=triplet_loss, metrics=["accuracy"])

#加载权值
load_weights_from_FaceNet(FRmodel)

end_time = time.clock()

minium = end_time - start_time
print("执行了："+str(int(minium/60)) + "分" + str(int(minium%60)))

"""构建一个数据库，里面包含了允许进入的人员的编码向量，注意这里要把images目录保存在当前目录下面，否则会报错，至于为什么我也没搞清楚"""
database = {}
database["danielle"] = img_to_encoding("images\\danielle.png", FRmodel)
database["younes"] = img_to_encoding("images\\younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images\\tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images\\andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images\\kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images\\dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images\\sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images\\bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images\\kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images\\felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images\\benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images\\arnaud.jpg", FRmodel)

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity"
    :param image_path: 摄像头的照片
    :param identity: 字符类型，想要验证的人的名字
    :param database: 字典类型，包含了成员的名字信息与对应的编码
    :param model: 在Keras的模型的实例
    :return:
            dist: 摄像头的图片与数据库中的图片的编码的差距( use the L2 distance (np.linalg.norm( ))
            door_open: 是否开门
    """
    ### Start code here ###

    # Step 1: 计算图像的编码
    encoding = img_to_encoding(image_path, model)
    # Step 2: 计算与数据库中保存的编码差距
    dist = np.linalg.norm(encoding - database[identity])
    # Step 3: 判断是否开门
    if dist < 0.7:
        print("It's " + str(identity) + ",welcome home!")
        door_open = True
    else:
        print("It,s not" + str(identity) + ",please go away")
        door_open = False
    ### End code here ###
    return dist, door_open
def test_verify():
    # younes来到门前，摄像头拍摄的照片存入camera_0.jpg，刷ID验证是否是younes本人
    dist, door_open = verify("images\\camera_0.jpg", "younes", database, FRmodel)
    print(dist, door_open)
    # Benoit拿着kian的ID来，刷kian的ID卡想进入房子，结果应该是验证失败
    dist1, door_open1 = verify("images\\camera_2.jpg", "kian", database, FRmodel)
    print(dist1, door_open1)
test_verify()
"""上面是人脸验证：数据库中存入人员的预先encoding，然后通过摄像头拍照，将图片进行encoding后，任何根据身份证的ID名来查询数据库中预先按存入的encoding
   计算编码距离，然后根据阈值来验证门前的这个人是否是本人，还是很简单，
   但是现在有个人的身份证被偷了，或者丢了，进不去家门了，是不是这是一种很糟糕的情况，那么人脸识别就来帮你解决这件麻烦事,那么人们就不需要带身份证，被授权的人只要站在门前，
   门就会自动打开，如下思路：
   step1：根据image_path计算图像的encoding
   step2：从数据库中找到与目标编码具有最小差距的编码
          ①初始化min_dist变量为足够大的数字(比如100)，它负责找到与输入编码最接近的编码。
          ②遍历数据库的名字和  oding，for (name, db_encoding) in database.items()
                ·计算目标编码与当前数据库编码之间的L2距离
                ·如果距离小于min_dist,那么就更新名字与编码到identity与min_dist中"""
def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    for (name, dic_enc) in database.items():
        dist = np.linalg.norm(encoding - dic_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) +", the distance is " + str(min_dist))
    return min_dist, identity
def test_whoisit():
    min_dist, identity = who_is_it("images\\camera_0.jpg", database, FRmodel)
    print(min_dist, identity)
    min_dist1, identity1 = who_is_it("images\\camera_1.jpg", database, FRmodel)
    print(min_dist1, identity1)
test_whoisit()
"""
请记住：
1·人脸验证解决了更容易的1:1匹配问题，人脸识别解决了更难的1∶k匹配问题。
2·triplet loss是训练神经网络学习人脸图像编码的一种有效的损失函数。
3·相同的编码可用于验证和识别。测量两个图像编码之间的距离可以确定它们是否是同一个人的图片。
"""

















































