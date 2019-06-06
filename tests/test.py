#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/6/5 13:11
@Author  : xumj
'''
import face_recognition
import os
import sys
import numpy as np
from click.testing import CliRunner

from face_recognition import api
from face_recognition import face_recognition_cli
from face_recognition import face_detection_cli
import dlib
# image = face_recognition.load_image_file("test_images/obama.jpg")
# face_locations = face_recognition.face_locations(image)
# face_landmarks_list = face_recognition.face_landmarks(image)

# known_image = face_recognition.load_image_file("test_images/400_0.jpg")
known_image = face_recognition.load_image_file("F:/faces/all_train/209/209_0.bmp")
# unknown_image = face_recognition.load_image_file("test_images/401_0.jpg")
unknown_image = face_recognition.load_image_file("F:/faces/all_train/209/209_1.bmp")
#
biden_encoding = face_recognition.face_encodings(known_image,face_recognition.face_locations(known_image,1,"cnn"))[0]
# biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image,face_recognition.face_locations(unknown_image,1,"cnn"))[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#
results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(results)

print(sys.getfilesystemencoding())
print("")



