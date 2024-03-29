# Find all the faces in an image then recognize them using a SVM with scikit-learn
# This allows you to train multiple images per person

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

# Install scikit-learn if you haven't already with pip
# $ pip3 install scikit-learn

import face_recognition
from sklearn import svm
import os

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []
# E:/workspace/pyProjects/xmj-project/face_recognition/examples/knn_examples/train
train_path,test_path,model_save_file="F:/faces/train/","F:/faces/test/","trained_svm_model_xmj.clf"
# train_path,test_path,model_save_file="E:/workspace/pyProjects/xmj-project/face_recognition/examples/knn_examples/train/","E:/workspace/pyProjects/xmj-project/face_recognition/examples/knn_examples/train/","trained_svm_model_xmj.clf"
# Training directory
train_dir = os.listdir(train_path)

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir(train_path + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file(train_path + person + "/" + person_img)
        if face_recognition.face_encodings(face).__len__() ==0:
            continue
        face_enc = face_recognition.face_encodings(face)[0]

        # Add face encoding for current image with corresponding label (name) to the training data
        encodings.append(face_enc)
        names.append(person)

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings, names)

# # Load the test image with unknown faces into a numpy array
# test_image = face_recognition.load_image_file('biden.jpg')
#
# # Find all the faces in the test image using the default HOG-based model
# face_locations = face_recognition.face_locations(test_image)
# no = len(face_locations)
# print("Number of faces detected: ", no)
#
# # Predict all the faces in the test image using the trained classifier
# print("Found: \n")
# for i in range(no):
#     test_image_enc = face_recognition.face_encodings(test_image)[i]
#     name = clf.predict([test_image_enc])
#     print(*name)

test_dir = os.listdir(test_path)
real=0
for person in test_dir:
    pix = os.listdir(test_path + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        # face = face_recognition.load_image_file(test_dir + person + "/" + person_img)
        # face_enc = face_recognition.face_encodings(face)[0]

        # Load the test image with unknown faces into a numpy array
        test_image = face_recognition.load_image_file(test_path + person + "/" + person_img)

        # Find all the faces in the test image using the default HOG-based model
        face_locations = face_recognition.face_locations(test_image)
        no = len(face_locations)
        print("Number of faces detected: ", no)

        # Predict all the faces in the test image using the trained classifier
        print("Found: \n")
        for i in range(no):
            test_image_enc = face_recognition.face_encodings(test_image)[i]
            name = clf.predict([test_image_enc])
            if str.find(person_img,name[i]) !=-1:
                real+=1
                print("True------------>{},name:{}".format(real,name))
