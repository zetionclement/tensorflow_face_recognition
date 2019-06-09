import cv2
import os
import re
import sys
import dlib
import facenet
import time
import align.detect_face
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import misc
from PIL import Image,ImageDraw,ImageFont
from multiprocessing import Process,Manager,Queue
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


detector = dlib.get_frontal_face_detector()    #先使用dlib检测人脸，后再使用卷积神经网络识别人脸
model_path="/home/boss/Study/face_recognition_flask/20180402-114759/20180402-114759.pb"
path_csv_feature_all="features_all_tensorflow.csv"
test_result_path="/home/boss/Study/face_recognition_flask/test_result_dlib"

success_list=[]   #保存已识别的人的名字
global csv_rd


#遍历本地features_all_tensorflow.csv文件中已保存的人脸数据，将所有人的特征存放到feature_known_list中
def known_faces(feature_known_list):
    f=open(path_csv_feature_all)
    global csv_rd
    csv_rd=pd.read_csv(f,header=None)
    for i in range(csv_rd.shape[0]):
        feature_someone_list=[]
        for j in range(1,len(csv_rd.ix[i,:])):
            feature_someone_list.append(csv_rd.ix[i,:][j])
        feature_known_list.append(feature_someone_list)
    print("数据库人脸数：", len(feature_known_list))
    return feature_known_list


def put_text(img_rd,text,position,fillcolor="#FF0000"):   #在摄像头上面打印信息
    img = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    img_PIL = Image.fromarray(img)
    font = ImageFont.truetype('NotoSansCJK-Black.ttc', 40, encoding="utf-8")
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, fillcolor, font)
    img = cv2.cvtColor(np.array(img_PIL),cv2.COLOR_RGB2BGR)
    return img


#裁剪人脸
def crop_image(image,bounding_boxes,margin,image_size):

    faces_queue=Queue()  #faces_queue存放裁剪下来的人脸
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

    emb_list=[]   #保存每一帧画面中所有人脸的特征
    faces_queue=crop_image(image,bounding_boxes,44,160)  #先裁剪人脸
    g=tf.get_default_graph()
    with g.as_default():
        sess=tf.get_default_session()
        with sess.as_default() :
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for i in range(faces_queue.qsize()):
                face=faces_queue.get()       #从faces_queue中取出已裁剪的人脸
                face=face.reshape(1,160,160,3)

                #计算人脸特征
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
    #print("欧式距离为: ", dist)
    return dist


#人脸识别
def face_recognition(image,bounding_boxes,feature_known_list,pos_namelist,name_namelist):

    del pos_namelist[:] # 人脸名字的坐标，每次用的时候先清空
    del name_namelist[:] # 人脸名字，每次用的时候先清空

    min_eculidean_position_list=[] #打印在人脸旁边的最小欧式距离的坐标


    features_cap_list = return_512D_features(image,bounding_boxes)   #features_cap_list为帧画面中所有人的人脸特征信息

    current_time = str(datetime.now())
    save_path = os.path.join(test_result_path, current_time)
    # 遍历捕获到的图像中所有的人脸
    for k in range(len(features_cap_list)):
        # 让人名跟随在矩形框的下方
        # 确定人名的位置坐标
        # 先默认所有人不认识
        name_namelist.append("未能识别")

        # 每个捕获人脸名字的坐标
        pos_namelist.append(
        tuple([bounding_boxes[k][0], int(bounding_boxes[k][3] + (bounding_boxes[k][3] - bounding_boxes[k][1]) / 15)]))

        # 每个捕获人脸最小欧士距离的坐标
        min_eculidean_position_list.append(tuple([bounding_boxes[k][0], int(pos_namelist[k][1]+50)]))

        person_euclidean_list=list()
        # 对于第k张人脸，遍历所有存储的人脸特征
        for i in range(len(feature_known_list)):
            #print("和本地数据第", str(i + 1), "个人相比， ", end='')
            # 将某张人脸与存储的所有人脸数据进行比对
            euclidean_dist  = return_euclidean_distance(features_cap_list[k], feature_known_list[i])
            person_euclidean_list.append(euclidean_dist)
        index=person_euclidean_list.index(min(person_euclidean_list))
        if person_euclidean_list[index] <=0.85:  # 即使找到一个最相似的脸，也要设定一个阀值（根据实际情况自行设定），只有低于这个阀值时才能认为是同一个人
            name_namelist[k] = str(csv_rd[0][index])
            cv2.rectangle(image,(bounding_boxes[k][0],bounding_boxes[k][1]),(bounding_boxes[k][2],bounding_boxes[k][3]),(0,255,255),3)  #在图片上用矩形框人脸
            image=put_text(image,str(csv_rd[0][index]),pos_namelist[k])     #在图片上打印名字
            image = put_text(image, str(round(person_euclidean_list[index],2)), min_eculidean_position_list[k])   #在图片上打印欧士距离
        else:
            cv2.rectangle(image, (bounding_boxes[k][0], bounding_boxes[k][1]),(bounding_boxes[k][2], bounding_boxes[k][3]), (0, 255, 255), 3)  #在图片上用矩形框人脸
            image=put_text(image,str(round(person_euclidean_list[index],2)),min_eculidean_position_list[k])    #在图片上打印欧士距离
    cv2.imwrite(save_path+".jpg",image)

    print("\n")
    print("屏幕中的人脸为:", name_namelist,"\n")


# 打开摄像头保存帧
def save_frame(images_que, pos_namelist, name_namelist, open_time):
    url = 'rtsp://admin:root123456@192.168.1.104:554//Streaming/Channels/1'       #这里采用的是海康威视的ip摄像头
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        f = open("info.txt", 'a')       #创建一个info.txt用于保存摄像头是否开启成功
        f.write("True\n")               #成功的话就写一个True进去
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
            if temp == 22:      #这里设定每22帧就保存1帧，如果每一帧都要进行人脸识别的话可能会卡顿，可以根据自己的实际情况设定
                #print("保存一帧")
                images_que.put(frame)
                #print("队列帧数为：%d" % (images_que.qsize()))
                temp = 0
        # 20分钟后自动关闭摄像头，可以自行设定，因为采用的是多进程，所以要逐一kill
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


# 人脸检测和人脸识别
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

            while True:
                image = images_que.get()   #从image_que取一张图片
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                #print("开始检测人脸")
                faces_coordinate = detector(img_gray, 0)        #检测人脸，faces_coordinate的形状为[n,4],n为边框数，即人脸数，4分别对应x1，y1，x2，y2
                faces_num=len(faces_coordinate)          #人脸数
                print("人脸数为：%d" % (faces_num))
                if faces_num != 0:
                    bounding_boxes = np.ndarray(shape=(faces_num, 4), dtype=np.int32)   #bounding_boxes存放每个人的人脸坐标
                    for k,d in enumerate(faces_coordinate):
                        bounding_boxes[k][0] = d.left()
                        bounding_boxes[k][1] = d.top()
                        bounding_boxes[k][2] = d.right()
                        bounding_boxes[k][3] = d.bottom()
                    face_recognition(image, bounding_boxes, feature_known_list, pos_namelist,
                                     name_namelist)  # 如果有人脸就调用人脸识别函数
                else:
                    print("\n")


#主进程
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
        images_que = Queue()         #用来保存从摄像头拍到的帧
        p1 = Process(target=save_frame, args=(images_que, pos_namelist, name_namelist,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),)) #进程1用于打开摄像头，并保存帧
        print("Create ProcessP1\n")
        p2 = Process(target=face_check,args=(images_que, feature_known_list, pos_namelist, name_namelist,))    #进程2用于人脸检测和人脸识别
        print("Create ProcessP2\n")
        p1.start()
        p2.start()
        p1.join()
        p2.join()

main_process()