from scipy import misc
import tensorflow as tf
import numpy as np
import os
import dlib
import pymysql
import shutil
import facenet
import cv2
import align.detect_face

detector = dlib.get_frontal_face_detector()    #先使用dlib检测人脸，后再使用卷积神经网络识别人脸
model_path="20180402-114759/model.pb"   #模型保存的路径
image_paths_uncalculated='data_faces_from_camera/Uncalculated/'   #Uncalculated文件夹下为未经过计算转化的人脸文件
image_paths_calculated='data_faces_from_camera/calculated/'       #calculated文件夹下为经过计算转化的人脸文件
db = pymysql.connect('127.0.0.1','root','123456','person_embeddings_db',charset='utf8mb4')
cursor = db.cursor()

#找到人脸以及给出人脸框数组
def load_and_align_data(image_path,image_size,margin): #margin为要剪裁的余量

    minisize=20 # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img = misc.imread(os.path.expanduser(image_path))  # 读取图片
    img_size = np.asarray(img.shape)[0:2]  # img[0]为宽度，img[1]为高度
    image_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces_coordinate = detector(image_gray,0)
    faces_num = len(faces_coordinate)
    bounding_boxes = np.ndarray(shape=(faces_num,4),dtype=np.int32)
    for k,d in enumerate(faces_coordinate):
        bounding_boxes[k][0] = d.left()
        bounding_boxes[k][1] = d.top()
        bounding_boxes[k][2] = d.right()
        bounding_boxes[k][3] = d.bottom()
    # bounding_boxes, _ = align.detect_face.detect_face(img, minisize, pnet, rnet, onet, threshold,
    #                                                   factor)  # 读取并对齐人脸，bounding_boxes为人脸框数组，形状为[n,5]，n代表边框数，这里一般只有1张人脸，5对应x1，y1，x2，y2，score，分别是左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标，人脸置信度
    det=np.squeeze(bounding_boxes[0,0:4])          #假设图片里的人脸数为1，所以这里要去除边框数那一维
    for i in range(bounding_boxes.shape[0]):
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


# 将uncalculated文件夹下面的人脸计算转化为512维的特征数组并写进以个人名字命名的表
def compute_feaure_and_write_into_person_database(image_paths_uncalculated,images_placeholder,embeddings,phase_train_placeholder):
    person_list=os.listdir(image_paths_uncalculated)
    print("6666")
    if len(person_list)>0: # uncalculated文件夹下有人脸文件才计算，否则输出没有人脸可计算
        for person in person_list:  # person为uncalculated中每个文件夹的名字，这里建议命名为人的名字
            sql = """CREATE TABLE IF NOT EXISTS %s (feature_1 DOUBLE NOT NULL""" % (person)   # 以人的名字创建一个表，里面存放每张图片的特征值
            for index in range(2, 513):
                temp = """,feature_%d DOUBLE NOT NULL""" %(index)
                sql += temp
            sql += """)ENGINE=InnoDB DEFAULT CHARSET=utf8;"""
            try:
                cursor.execute(sql)
                db.commit()
            except:
                db.rollback()
                print("创建数据表——%s失败！"%(person))
                continue

            path_faces_personX=image_paths_uncalculated+person  # path_faces_personX为每个人的人脸文件夹
            images=os.listdir(path_faces_personX)   # 列出每个人脸文件夹下面的图片
            for i in range(len(images)):
                if os.path.exists(path_faces_personX + "/" + images[i]):
                    print("正在读的人脸图像：", path_faces_personX + "/" + images[i])
                    feature_512D=return_512D_features(path_faces_personX+"/"+images[i],images_placeholder,embeddings,phase_train_placeholder) #计算每张人脸图片，返回第i张图片的特征数组
                    feature_512D=np.squeeze(feature_512D)    # 将feature_512D转化为1维数组
                    feature_512D=list(feature_512D)
                    if len(feature_512D)<1:    # feature_512D长度小于1可能是没检测到人脸，所以跳过
                        i+=1
                    else:
                        sql = """INSERT INTO %s (feature_1"""%(person)   # 将经过神经网络计算得到的人脸特征值feature_512D存放到对应的表中
                        for index in range(2, 513):
                            temp = """,feature_%d""" % (index)
                            sql += temp
                        sql += """) VALUES(\'%.20f\'""" % (feature_512D[0])
                        for index in range(1, 512):
                            sql += """,\'%.20f\'""" % (feature_512D[index])
                        sql += """);"""
                    try:
                        cursor.execute(sql)
                        db.commit()
                    except:
                        db.rollback()
                        print("%s数据表插入失败"%(person))
                        continue

            shutil.move(image_paths_uncalculated+person,image_paths_calculated)  #当一个人的所有图片被计算完后，就把这个人的人脸文件夹移到calculated文件夹下
    else:
        print("没有人脸可计算")


# 计算所有人的人脸特征均值并存放到person_average_embeddings表中
def compute_the_mean_and_insert_into_person_average_embeddings():

    sql = """CREATE TABLE IF NOT EXISTS person_average_embeddings    # 如果没有person_average_embeddings这个表，就创建一个
             (name VARCHAR(255) UNIQUE NOT NULL"""
    for index in range(1, 513):
        temp = """,feature_%d DOUBLE NOT NULL""" % (index)
        sql += temp
    sql += """)ENGINE=InnoDB DEFAULT CHARSET=utf8;"""
    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()
        print("create table failed!")


    sql = """SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA='person_embeddings_db';"""
    cursor.execute(sql)
    person_name_tuple = cursor.fetchall()
    for i in range(len(person_name_tuple)):
        if person_name_tuple[i][0] == "person_average_embeddings":
            continue
        else:
            sql = """SELECT feature_1"""
            for index in range(2, 513):
                temp = """,feature_%d""" % (index)
                sql += temp
            sql += """ FROM person_embeddings_db.%s""" % (person_name_tuple[i][0])
            cursor.execute(sql)
            embeddings_tuple = cursor.fetchall()
            embeddings_array = np.array(embeddings_tuple)
            embeddings_array_mean = np.mean(embeddings_array, axis=0).reshape(-1)
            sql = """INSERT INTO person_average_embeddings (name"""
            for index in range(1, 513):
                temp = """,feature_%d""" % (index)
                sql += temp
            sql += """) VALUES(\'%s\'""" % (person_name_tuple[i][0])
            for index in range(0,512):
                sql += """,\'%.20f\'""" % (embeddings_array_mean[index])
            sql += """);"""
            try:
                cursor.execute(sql)
                db.commit()
            except:
                db.rollback()
                print("Insert into person_average_embeddings failed!")



if __name__=="__main__":
    with tf.Graph().as_default():
        config=tf.ConfigProto()
        config.gpu_options.allocator_type="BFC"
        sess=tf.Session(config=config)
        with sess.as_default():
            # pnet,rnet,onet=align.detect_face.create_mtcnn(sess,None)   #加载MTCNN的3层网络，用来检测人脸
            #
            # # 加载卷积神经网络模型
            facenet.load_model(model_path)     #加载人脸识别模型，用来识别人脸
            # 获取输入和输出张量
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            compute_feaure_and_write_into_person_database(image_paths_uncalculated,images_placeholder,embeddings,phase_train_placeholder) #计算uncalculated文件夹下面所有人的人脸特征，将每个人的人脸特征存放到path_csv_feature中
            compute_the_mean_and_insert_into_person_average_embeddings()