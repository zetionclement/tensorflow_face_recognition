from scipy import misc
import tensorflow as tf
import numpy as np
import pandas as pd
import dlib
import os
import csv
import cv2
import shutil
import facenet
import align.detect_face

model_path="20180402-114759/20180402-114759.pb"   #模型保存的路径
image_paths_uncalculated='data_faces_from_camera/Uncalculated/'   #Uncalculated文件夹下为未经过计算转化的人脸文件
image_paths_calculated='data_faces_from_camera/calculated/'       #calculated文件夹下为经过计算转化的人脸文件
path_csv_feature="data_csvs_from_camera/" #存放每个人的人脸的csv
path_csv_feature_all="features_all_tensorflow.csv" #存放全部人的人脸特征
detector = dlib.get_frontal_face_detector()

#找到人脸以及给出人脸框数组
def load_and_align_data(image_path,image_size,margin): #margin为要剪裁的余量

    minisize=20 # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img = misc.imread(os.path.expanduser(image_path))  # 读取图片
    img_size = np.asarray(img.shape)[0:2]  # img[0]为宽度，img[1]为高度
    bounding_boxes, _ = align.detect_face.detect_face(img, minisize, pnet, rnet, onet, threshold,
                                                      factor)  # 读取并对齐人脸，bounding_boxes为人脸框数组，形状为[n,5]，n代表边框数，这里一般只有1张人脸，5对应x1，y1，x2，y2，score，分别是左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标，人脸置信度
    det=np.squeeze(bounding_boxes[0,0:4])          #假设图片里的人脸数为1，所以这里要去除边框数那一维
    bb=np.zeros(4,dtype=np.int32)            #bb为要剪裁的人脸区域
    bb[0]=np.maximum(det[0]-margin/2,0)      #左上角x1
    bb[1]=np.maximum(det[1]-margin/2,0)      #左上角y1
    bb[2]=np.minimum(det[2]+margin/2,img_size[1])      #右下角x2
    bb[3]=np.minimum(det[3]+margin/2,img_size[0])      #右下角y2
    cropped=img[bb[1]:bb[3],bb[0]:bb[2],:]    #根据bb来裁剪原图片中的人脸
    aligned=misc.imresize(cropped,(image_size,image_size),interp='bilinear')   #将图片缩放为卷积神经网络模型输入的大小
    prewhitened=facenet.prewhiten(aligned)     #对裁剪出的人脸进行图片标准化处理

    return prewhitened


#计算并返回人脸特征
def return_512D_features(image_path,images_placeholder,embeddings,phase_train_placeholder):

    images=load_and_align_data(image_path,160,44)    #使用MTCNN算法找到图片中的人脸并给出人脸框数组
    images=images.reshape(1,160,160,3)

    feed_dict={images_placeholder:images,phase_train_placeholder:False}
    emb=sess.run(embeddings,feed_dict=feed_dict)     #计算人脸特征，得到人脸特征数组
    return emb

#将uncalculated文件夹下面的人脸计算转化为512维的特征数组并写进个人csv文件
def compute_feaure_and_write_into_csv(image_paths_uncalculated,images_placeholder,embeddings,phase_train_placeholder):
    person_list=os.listdir(image_paths_uncalculated)
    if len(person_list)>0: #uncalculated文件夹下有人脸文件才计算，否则输出没有人脸
        for person in person_list:  #person为uncalculated中每个文件夹的名字，这里建议命名为人的名字
            print(path_csv_feature+person+".csv")
            path_csv=path_csv_feature+person+".csv"  #path_csv为以某人的名字命名的csv文件的路径
            path_faces_personX=image_paths_uncalculated+person  #path_faces_personX为每个人的人脸文件夹
            images=os.listdir(path_faces_personX)   #列出每个人脸文件夹下面的图片
            with open(path_csv,"w",newline="") as csvfile:
                writer=csv.writer(csvfile)      #创建一个writer对象
                for i in range(len(images)):
                    if os.path.exists(path_faces_personX + "/" + images[i]):
                        print("正在读的人脸图像：", path_faces_personX + "/" + images[i])
                        feature_512D=return_512D_features(path_faces_personX+"/"+images[i],images_placeholder,embeddings,phase_train_placeholder) #计算每张人脸图片，返回第i张图片的特征数组
                        feature_512D=np.squeeze(feature_512D)    #将feature_512D转化为1维数组
                        feature_512D=list(feature_512D)
                        if len(feature_512D)<1:    #feature_512D长度小于1可能是没检测到人脸，所以跳过
                            i+=1
                        else:
                            writer.writerow(feature_512D)  #将feature_512D写进csv文件
            shutil.move(image_paths_uncalculated+person,image_paths_calculated)  #当一个人的所有图片被计算完后，就把这个人的人脸文件夹移到calculated文件夹下
    else:
        print("没有人脸可计算")


def compute_the_mean_and_write_into_all_csv(path_csv_feature):
    with open(path_csv_feature_all,"w",newline="")as csvfile:
        writer=csv.writer(csvfile)
        csv_rd=os.listdir(path_csv_feature)   #csv_rd为path_csv_feature下所有人的csv文件

        for i in range(len(csv_rd)):
            path_csv_rd=path_csv_feature+csv_rd[i]   #path_csv_rd为每个人的csv文件的路径
            column_names=[]     #给下面要读取的每个人的csv文件起列名
            for feature_num in range(512):
                column_names.append("features_" + str(feature_num + 1))

            f=open(path_csv_rd)     #打开某人的csv文件
            rd=pd.read_csv(f,names=column_names)  #读取某人的csv文件，制定列名为column_names

            feature_mean=[]   #feature_mean为将要写入features_all_tensorflow.csv的列表，一共513列，第1列为名字，剩余512列为特征值

            name=path_csv_rd.split('/')[1].split('.')[0]  #因为csv文件是以人的名字命名的，所以直接从文件名中获取人的名字
            feature_mean.append(name)     #第一列存入人的名字
            for feature_num in range(512):
                tmp_arr=rd["features_" + str(feature_num + 1)]    #从某人的csv文件中读取第feature_num列，对应的列名为feature_num + 1
                tmp_arr=np.array(tmp_arr)       #转化为array

                tmp_mean=np.mean(tmp_arr)       #计算那一列的特征均值
                feature_mean.append(tmp_mean)   #将那一列加入到feature_mean中
            writer.writerow(feature_mean)  #读取完512列后就写入到features_all_tensorflow.csv



if __name__=="__main__":
    with tf.Graph().as_default():
        config=tf.ConfigProto()
        config.gpu_options.allocator_type="BFC"
        sess=tf.Session(config=config)
        with sess.as_default():
            pnet,rnet,onet=align.detect_face.create_mtcnn(sess,None)   #加载MTCNN的3层网络，用来检测人脸

            # Load the model
            facenet.load_model(model_path)     #加载人脸识别模型，用来识别人脸
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            compute_feaure_and_write_into_csv(image_paths_uncalculated,images_placeholder,embeddings,phase_train_placeholder) #计算uncalculated文件夹下面所有人的人脸特征，将每个人的人脸特征存放到path_csv_feature中
            compute_the_mean_and_write_into_all_csv(path_csv_feature)