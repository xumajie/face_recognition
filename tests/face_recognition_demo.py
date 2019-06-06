#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/6/6 15:18
@Author  : xumj
测试 300张图片的识别率
250图片有，50张没有
调用dlib人脸特征编码直接识别
'''
import os
import re
import face_recognition
from face_recognition import api

train_total,test_total=10,15

# 获取该文件夹下面的图片
def image_file_in_folder(folder):
    return [os.path.join(folder,f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|bmp)',f,flags=re.I)]


def get_face_compare_list(train_dir):
    train_index=0
    faces_to_compare = []
    for class_dir in os.listdir(train_dir):
        # 跳过目录
        if not os.path.isdir(os.path.join(train_dir,class_dir)):
            continue
        for img_path in image_file_in_folder(os.path.join(train_dir,class_dir)):
            if img_path.find('4.bmp') >= 0:
                continue
            image=face_recognition.load_image_file(img_path)
            face_bounding_boxes=face_recognition.face_locations(image)
            # face_bounding_boxes=face_recognition.face_locations(image,model="cnn")
            # 简单不见识别特征，使用cnn模型
            if len(face_bounding_boxes)== 0:
                face_bounding_boxes=face_recognition.face_locations(image,1,"cnn")

            if len(face_bounding_boxes)!=1:
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                faces_to_compare.append({"name":img_path,"face_encodings":face_recognition.face_encodings(image,known_face_locations=face_bounding_boxes)[0]})
        train_index=train_index+1
        if train_index>=train_total:
            return faces_to_compare
#     获取最相似图片，如果对比所有图片阀值小于 0.6则判断没有相似读片
def get_face_name(faces_to_compare,face_encodings):
    orgin_name,targe_name,simily='','',0
    for o in faces_to_compare:
        name=o["name"]
        _face_encodings=o["face_encodings"]
        max_distance=max(api.face_distance([face_encodings],_face_encodings))
        if max_distance > 0.8 :
            return name,max_distance
        else  :
            simily= simily if simily >max_distance else max_distance
            targe_name= targe_name if simily >max_distance else name
    return targe_name,simily

def get_face_regconge_result(faces_to_compare,train_dir):
    name_list=[]
    test_index=0
    for class_dir in os.listdir(list[0]):
        # 跳过目录
        if not os.path.isdir(os.path.join(train_dir,class_dir)):
            continue
        for img_path in image_file_in_folder(os.path.join(train_dir,class_dir)):
            if img_path.find('4.bmp') ==-1:
                continue
            image=face_recognition.load_image_file(img_path)
            # face_bounding_boxes=face_recognition.face_locations(image,model="cnn")
            face_bounding_boxes=face_recognition.face_locations(image)
            # 简单不见识别特征，使用cnn模型
            if len(face_bounding_boxes)==0:
                face_bounding_boxes=face_recognition.face_locations(image,1,"cnn")

            if len(face_bounding_boxes)!=1:
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                face_encodings=face_recognition.face_encodings(image,known_face_locations=face_bounding_boxes)[0]
                name_list.append({"img":img_path,"target":get_face_name(faces_to_compare,face_encodings)})

        test_index+=1
        if test_index==test_total:
            return name_list

if __name__ == '__main__':
    train_dir = "F:/face"
    s = "F:/faces/all/,F:/faces/bak/test/,trained_knn_model_xmj_11_test.clf,2"
    list = str.split(s, ",")
    faces_to_compare=get_face_compare_list(list[0])
    results=get_face_regconge_result(faces_to_compare,list[0])
    print(results)
