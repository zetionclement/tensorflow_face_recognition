import cv2
import os
import re
import sys
import time
import facenet
import align.detect_face
import numpy as np
import pandas as pd
from scipy import misc
from PIL import Image,ImageDraw,ImageFont
from multiprocessing import Process,Manager,Queue
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model_path="20180402-114759/20180402-114759.pb"
path_csv_feature_all="features_all_tensorflow.csv"
cascPath="FaceDetect-master/haarcascade_frontalface_default.xml"

success_list=[]
global csv_rd


def known_faces(feature_known_list):
    f=open(path_csv_feature_all)
    global csv_rd
    csv_rd=pd.read_csv(f,header=None)
    for i in range(csv_rd.shape[0]):
        feature_someone_list=[]
        for j in range(1,len(csv_rd.ix[i,:])):
            feature_someone_list.append(csv_rd.ix[i,:][j])
        feature_known_list.append(feature_someone_list)
    print("Faces in Database：", len(feature_known_list))
    return feature_known_list


def put_text(img_rd,text,position,fillcolor="#FF0000"):
    img = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    img_PIL = Image.fromarray(img)
    font = ImageFont.truetype('SIMYOU.TTF', 40, encoding="utf-8")
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, fillcolor, font)
    img = cv2.cvtColor(np.array(img_PIL),cv2.COLOR_RGB2BGR)
    return img


def crop_image(image,bounding_boxes,margin,image_size):

    faces_queue=Queue()
    img_size = np.asarray(image.shape)[0:2]
    for i in range(bounding_boxes.shape[0]):
        bb=np.zeros(4,dtype=np.int32)
        bb[0]=np.maximum(bounding_boxes[i][0]-margin/2,0)
        bb[1]=np.maximum(bounding_boxes[i][1]-margin/2,0)
        bb[2]=np.minimum(bounding_boxes[i][2]+margin/2,img_size[1])
        bb[3]=np.minimum(bounding_boxes[i][3]+margin/2,img_size[0])
        cropped=image[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned=misc.imresize(cropped,(image_size,image_size),interp='bilinear')
        prewhitened=facenet.prewhiten(aligned)
        faces_queue.put(prewhitened)
    return faces_queue


def return_512D_features(image,bounding_boxes):

    emb_list=[]
    faces_queue=crop_image(image,bounding_boxes,44,160)
    g = tf.get_default_graph()
    with g.as_default():
        sess = tf.get_default_session()
        with sess.as_default():
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for i in range(faces_queue.qsize()):
                face=faces_queue.get()
                face=face.reshape(1,160,160,3)


                feed_dict={images_placeholder:face,phase_train_placeholder:False}
                emb=sess.run(embeddings,feed_dict=feed_dict)
                emb=list(np.squeeze(emb))
                emb_list.append(emb)
            return emb_list


# 计算两个人脸向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(np.subtract(feature_1,feature_2))))
    print("欧式距离为: ", dist)
    return dist



def face_recognition(image,bounding_boxes,feature_known_list,pos_namelist,name_namelist):

    del pos_namelist[:]
    del name_namelist[:]

    features_cap_list = return_512D_features(image,bounding_boxes)

    # 遍历捕获到的图像中所有的人脸
    for k in range(len(features_cap_list)):
        # 让人名跟随在矩形框的下方
        # 确定人名的位置坐标
        # 先默认所有人不认识
        name_namelist.append("未能识别")
        '''
        # 每个捕获人脸名字的坐标
        pos_namelist.append(
        tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 7)]))
        '''
        person_euclidean_list=list()
        # 对于第k张人脸，遍历所有存储的人脸特征
        for i in range(len(feature_known_list)):
            print("和本地数据第", str(i + 1), "个人相比， ", end='')
            # 将某张人脸与存储的所有人脸数据进行比对
            euclidean_dist = return_euclidean_distance(features_cap_list[k], feature_known_list[i])
            person_euclidean_list.append(euclidean_dist)
            index = person_euclidean_list.index(min(person_euclidean_list))
            if person_euclidean_list[index] <= 1.0:  # 即使找到一个最相似的脸，也要设定一个阀值（根据实际情况自行设定），只有低于这个阀值时才能认为是同一个人
                global csv_rd
                name_namelist[k] = str(csv_rd[0][index])
    print("\n")
    print("屏幕中的人脸为:", name_namelist,"\n")



def save_frame(images_que, pos_namelist, name_namelist, open_time):
    url = 'rtsp://admin:root123456@192.168.1.104:554//Streaming/Channels/1'
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        f = open("info.txt", 'a')
        f.write("True\n")
        f.close()
    temp = 0
    '''
    pid1 = os.getpid()
    f = open("info.txt", 'a')
    f.write('p1:' + str(pid1) + "\n")
    f.close()
    '''
    while True:
        ret, frame = cap.read()

        if ret:
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 1280, 720)
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            temp += 1
            if temp == 22:
                #print("保存一帧")
                images_que.put(frame)
                #print("队列帧数为：%d" % (images_que.qsize()))
                temp = 0
        # 20分钟后自动关闭摄像头
        if time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 1200)) >= open_time:
            f = open("info.txt")
            info_list = f.readlines()
            f.close()
            flask_temp = 0
            for i in range(len(info_list)):
                temp = re.findall('\d+$', info_list[i])
                if temp:
                    flask_temp = temp[0]
            pid_list = os.popen("ps -ef | grep flask").readlines()
            for i in range(len(pid_list)):
                pid_list[i] = pid_list[i].split()[1]
                if str(pid_list[i]) != flask_temp and flask_temp != 0:
                    try:
                        os.popen("sudo kill -15 " + str(pid_list[i]))
                    except:
                        os.popen("sudo kill -9 " + str(pid_list[i]))
                    print("kill " + str(pid_list[i]) + "\n")
            if os.path.exists("info.txt"):
                os.remove("info.txt")
            time.sleep(4)
            sys.exit()


def face_check(images_que, feature_known_list, pos_namelist, name_namelist):
    '''
    pid2 = os.getpid()
    f = open("info.txt", 'a')
    f.write('p2:' + str(pid2) + "\n")
    f.close()
    '''
    with tf.Graph().as_default():

        with tf.Session()as sess:
            # Load the model
            facenet.load_model(model_path)
            faceCascade=cv2.CascadeClassifier(cascPath)
            while True:
                image = images_que.get()
                #print("开始检测人脸")
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                faces_coordinate=faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE) #使用opencv的方法检测人脸
                faces_num=int(format(len(faces_coordinate)))
                print("人脸数为：%d" % (faces_num))
                if faces_num!=0:
                    bounding_boxes=np.ndarray(shape=(faces_num,4),dtype=np.int32)
                    i=0
                    for (x,y,w,h) in faces_coordinate:    #x代表左上角横坐标，y代表左上角纵坐标，w代表宽度，h代表高度
                        x2=x+w
                        y2=y+h
                        bounding_boxes[i][0]=x
                        bounding_boxes[i][1]=y
                        bounding_boxes[i][2]=x2
                        bounding_boxes[i][3]=y2
                        i+=1
                    face_recognition(image,bounding_boxes, feature_known_list, pos_namelist, name_namelist)  # 如果有人脸就调用人脸识别函数
                else:
                    print("\n")


def main_process():
    '''
    p=os.getpid()
    f=open("info.txt",'w')
    f.write('p:'+str(p)+"\n")
    f.close()
    '''
    with Manager() as manager:
        feature_known_list = manager.list()  # 已知的人脸的特征list
        pos_namelist = manager.list()  # 要在屏幕上打印的人脸名字的坐标
        name_namelist = manager.list()  # 要在屏幕上打印的人脸名字
        feature_known_list=known_faces(feature_known_list)  # 遍历所有已知的人脸数据
        images_que = Queue()              #用来保存从摄像头拍到的帧
        p1 = Process(target=save_frame, args=(images_que, pos_namelist, name_namelist,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),))
        print("Create ProcessP1\n")
        p2 = Process(target=face_check,args=(images_que, feature_known_list, pos_namelist, name_namelist,))
        print("Create ProcessP2\n")
        p1.start()
        p2.start()
        p1.join()
        p2.join()

main_process()