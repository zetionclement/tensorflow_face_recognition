# coding:utf-8
import cv2
import dlib
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from multiprocessing import Process,Manager,Queue


success_list=[]

# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data_dlib/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1("data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 存放所有人脸特征的 CSV
path_features_known_csv = "features_all.csv"
f = open(path_features_known_csv)
global csv_rd
csv_rd = pd.read_csv(f, header=None)

def put_text(img_rd,text,position,fillcolor="#FF0000"):
    img = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    img_PIL = Image.fromarray(img)
    font = ImageFont.truetype('SIMYOU.TTF', 40, encoding="utf-8")
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, fillcolor, font)
    img = cv2.cvtColor(np.array(img_PIL),cv2.COLOR_RGB2BGR)
    return img


# 计算两个人脸向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


# 遍历已保存的人脸
def known_faces(csv_rd,features_known_arr):
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(1, len(csv_rd.ix[i, :])):
            features_someone_arr.append(csv_rd.ix[i, :][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))


#人脸识别
def face_recognition(faces,img_rd,features_known_arr,pos_namelist,name_namelist):

    del pos_namelist[:] # 人脸名字的坐标
    del name_namelist[:] # 人脸名字

    features_cap_arr = [] # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
    for i in range(len(faces)):
        shape = predictor(img_rd, faces[i])  #输入原图和人脸坐标计算得到人脸特征值
        features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

    # 遍历捕获到的图像中所有的人脸
    for k in range(len(faces)):
        # 让人名跟随在矩形框的下方
        # 确定人名的位置坐标
        # 先默认所有人不认识
        name_namelist.append("未能识别")

        # 每个捕获人脸名字的坐标
        pos_namelist.append(
            tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 7)]))

        person_euclidean_list = list()
        # 对于第k张人脸，遍历所有存储的人脸特征
        for i in range(len(features_known_arr)):
            #print("with person_", str(i + 1), "the ", end='')
            # 将某张人脸与存储的所有人脸数据进行比对
            euclidean_dist = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
            person_euclidean_list.append(euclidean_dist)
            index = person_euclidean_list.index(min(person_euclidean_list))
            if person_euclidean_list[index] <= 0.7:  # 即使找到一个最相似的脸，也要设定一个阀值（根据实际情况自行设定），只有低于这个阀值时才能认为是同一个人
                global csv_rd
                name_namelist[k] = str(csv_rd[0][index])

    #print("屏幕中的人脸为:", name_namelist,"\n")


#在屏幕上打印人脸矩形框和人脸名字
def print_faces_pos(img_rd,faces_dict,pos_namelist,name_namelist):
    if len(faces_dict['faces'])>0:
        # 绘制矩形框
        for kk, d in enumerate(faces_dict['faces']):
            cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255),2)

        if len(pos_namelist)>0 and len(name_namelist)>0:
            # 写人脸名字
            for i in range(len(faces_dict['faces'])):
                img_rd = put_text(img_rd, name_namelist[i], pos_namelist[i], "#FF0000")
    return img_rd


#打开摄像头保存帧的函数
def save_frame(faces_que,faces_dict,pos_namelist,name_namelist,open_time):
    url = 'rtsp://admin:root123456@192.168.1.104:554//Streaming/Channels/1'
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        f=open("info.txt",'a')
        f.write("True\n")
        f.close()
    temp=0
    '''
    pid1 = os.getpid()
    f = open("info.txt", 'a')
    f.write('p1:' + str(pid1) + "\n")
    f.close()
    '''
    while True:
        ret,frame=cap.read()
        #frame=print_faces_pos(frame,faces_dict,pos_namelist,name_namelist)

        if ret:
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 1280, 720)
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            temp+=1
            if temp==22:
                #print("保存一帧")
                faces_que.put(frame)
                #print("队列帧数为：%d" % (faces_que.qsize()))
                temp=0
        # 20分钟后自动关闭摄像头
        if time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()-1200))>=open_time:
            f=open("info.txt")
            info_list=f.readlines()
            f.close()
            flask_temp=0
            for i in range(len(info_list)):
                temp=re.findall('\d+$',info_list[i])
                if temp:
                    flask_temp=temp[0]
            pid_list=os.popen("ps -ef | grep flask").readlines()
            for i in range(len(pid_list)):
                pid_list[i]=pid_list[i].split()[1]
                if str(pid_list[i])!=flask_temp and flask_temp!=0:
                    try:
                        os.popen("sudo kill -15 "+str(pid_list[i]))
                    except:
                        os.popen("sudo kill -9 " + str(pid_list[i]))
                    print("kill "+str(pid_list[i])+"\n")
            if os.path.exists("info.txt"):
                os.remove("info.txt")
            time.sleep(4)
            sys.exit()



#定义人脸检测的函数
def face_check(faces_que,features_known_arr,faces_dict,pos_namelist,name_namelist):
    '''
    pid2 = os.getpid()
    f = open("info.txt", 'a')
    f.write('p2:' + str(pid2) + "\n")
    f.close()
    '''
    while True:
        img_rd = faces_que.get()
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
        #print("开始检测人脸")
        faces = detector(img_gray, 0)  #faces为人脸坐标
        faces_dict['faces']=faces

        for k ,d in enumerate(faces):
            print(d.left(),d.top(),d.right(),d.bottom())
        print("人脸数为：%d\n" % (len(faces)))
        if len(faces) != 0:   # 检测到人脸
            face_recognition(faces, img_rd,features_known_arr,pos_namelist,name_namelist) #如果有人脸就调用人脸识别函数


#主进程
def main_process():
    '''
    p=os.getpid()
    f=open("info.txt",'w')
    f.write('p:'+str(p)+"\n")
    f.close()
    '''
    with Manager() as manager:
        features_known_arr=manager.list() #已知的人脸的特征list
        pos_namelist=manager.list()  #要在屏幕上打印的人脸名字的坐标
        name_namelist=manager.list()  #要在屏幕上打印的人脸名字
        faces_dict = manager.dict()  # 要在屏幕上打印的人脸矩形框坐标
        faces = dlib.rectangles()
        faces_dict['faces'] = faces
        known_faces(csv_rd,features_known_arr) #遍历所有已知的人脸数据
        faces_que=Queue()            #用来保存从摄像头拍到的帧
        p1=Process(target=save_frame,args=(faces_que,faces_dict,pos_namelist,name_namelist,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),))
        print("Create ProcessP1\n")
        p2=Process(target=face_check,args=(faces_que,features_known_arr,faces_dict,pos_namelist,name_namelist,))
        print("Create ProcessP2\n")
        p1.start()
        p2.start()
        p1.join()
        p2.join()

main_process()
