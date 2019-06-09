# tensorflow_face_recognition
使用tensorflow+opencv或tensorflow+mtcnn或dlib实现人脸识别

运行环境：

Tensorflow 1.11

Linux Ubuntu 18.04

### 使用前清先到模型原作者的github仓库中下载20180402-114759这个已经训练好的卷积模型文件，地址为https://github.com/davidsandberg/facenet ，下载完后解压到本项目当前路径下，这步至关重要。

#### 本项目使用的是海康威视的ip摄像头，并且将人脸识别分成两个部分来做，第一是人脸检测，第二是人脸识别，整个识别流程是首先在摄像头中找到人脸，如果有人脸就使用人脸识别模型，没有的话就继续寻找人脸，只有在画面中人脸数量大于0才调用人脸识别模型。

#### 本项目一共有四种解决方案，第一是dlib检测人脸，tensorflow模型识别人脸，对应的脚本文件名为face_recognition_dlib_tensorflow_people.py，第二是opencv检测人脸，tensorflow模型识别人脸，对应的脚本文件为face_recognition_opencv_tensorflow.py，第三种是mtcnn模型检测人脸，tensorflow模型识别人脸，对应的脚本文件为face_recognition_mtcnn_tensorflow_people.py，第四中是dlib检测人脸，dlib识别人脸，对应的脚本文件为face_recognition_dlib.py，经实际测试发现，使用第一种的效果最好，第二第三种的人脸检测成功率较低，第四种因为年代久远，所以不推荐使用。

#### 无论是哪种方案，都是用到了主进程创建进程1和进程2的策略，进程1负责打开摄像头，并且设定每隔22帧画面（测试用的海康威视ip摄像头为每秒22帧，每一秒保存一帧是为了防止卡顿）就保存一帧画面到Queue中（这个Queue可用于进程间的通信）用于人脸检测和识别，进程2负责从Queue中取出保存的帧，首先调用人脸检测模型检测人脸，如果人脸数大于零，就调用人脸识别模型进行人脸识别。代码运行的时候可以选择是否在控制台打印检测到的人脸数目，Queue保存的帧数等。此外，还设置了20分钟后自动关闭摄像头的功能，简而言之就是kill掉所有与之有关的进程。

#### 下面以第一种解决方案为例，介绍如何使用海康威视的ip摄像头进行人脸识别，其他方案原理类似，具体的说明代码里面都有注释，不再一一详细展开介绍

#### 1.首先请得到若干张(建议5张以上)同一个人的正脸照片，分别使用\*.jpg命名，\*为阿拉伯数字，从1开始，逐渐递增。

#### 2.将第一步得到的照片放进一个文件夹里面，文件夹的名字最好为照片本人的名字，然后把整个文件夹存放到data_faces_from_camera/Uncalculated/目录下

#### 3.运行get_features_into_csv_tensorflow.py脚本文件，这个脚本用于将上面步骤中的人脸录入到本地中保存为.csv文件，首先把每个人的每张照片转换为一个shape=[1,512]的数组（相当于将人脸信息数组化，一张人脸被转化成一个512D的特征数组），然后保存到data_csvs_from_camera中以每个人名字命名的.csv文件中，这个.csv文件存放的是将每个人的照片转换为数组的结果，有多少张照片就有多少行。举个例子，例如有一个人，名字叫1，他一共有7张照片，将他的7张照片按顺序命名好后放到一个叫1的文件夹，然后将文件夹1放到data_faces_from_camera/Uncalculated/目录下，因此在运行get_features_into_csv_tensorflow.py脚本文件后会在data_csvs_from_camera有一个名叫1.csv的文件，里面有7行数组，每一行都有512列，每一行代表每张图片转化后的结果。每计算完一个人后就自动把该人对应的人脸文件夹移动到目录data_faces_from_camera/calculated中。
#### 待所有人都将图片转换成.csv文件后，get_features_into_csv_tensorflow.py还会将每个人的.csv文件计算平均值，然后存放到features_all_tensorflow.csv文件中。假如说有1和2两个人，他们的人脸都被转换成1.csv文件和2.csv文件，那么接下来就会把1.csv文件和2.csv文件中的每一列计算平均值，存放到features_all_tensorflow.csv文件中，features_all_tensorflow.csv中存放着n个人的人脸信息（n等于data_csvs_from_camera中.csv文件的数量，即录入的人数）。features_all_tensorflow.csv文件一共有n行，每一行512列，每一行代表一个人的特征。

#### 4.然后运行face_recognition_dlib_tensorflow_people.py脚本进行人脸识别，运行前请先到脚本文件里面修改第158行的摄像头url，将摄像头的信息设置为自己摄像头的信息，这里用的是opencv打开摄像头的方法，详细使用方法可以百度一下，注意，不设置这步将无法正确运行。

#### 5.这里采用的人脸识别方法是如果画面中有人脸，就把画面中的所有人脸裁剪出来，放到一个faces_queue中，然后从faces_queue中取出上面裁剪的人脸输入到卷积神经网络中，卷积神经网络输出一个和人脸录入时一样的shape=[1,512]的数组，即人脸特征数组，将这个得到的人脸特征数组与本地中已录入的人脸特征数组两两之间计算欧几里德距离，取其中最小的欧几里德距离作为最后结果(距离越小代表两个人的特征越接近)，若欧士距离小于0.7，就认为是同一个人，并且将对应的人名打印在控制台上，画面中有多少张人脸就打印多少条名字，若果大于0.7，控制台就会输出“未能识别”。0.7是一个阈值，可以根据自己的实际情况调整。

#### 6.在face_recognition_dlib_tensorflow_people.py和face_recognition_mtcnn_tensorflow_people.py中还增加了标记人脸和自动保存图片的功能，代码会把能检测到的人脸（有的人脸可能检测不到）用黄色的矩形框起来，在矩形框下面会有对应的名字和最小的欧士距离，在face_recognition_mtcnn_tensorflow_people.py中还会在矩形框的上面打印人脸置信度（即这个矩形框里面是人脸的概率），这个人脸置信度只有大于0.85（可自行调节）才认为这是一张人脸。标记完所有人脸后就以当前时间命名并保存该图片到test_result_dlib文件夹中，里面有一张图片是测试后的结果，face_recognition_mtcnn_tensorflow_people.py脚本的保存在test_result_mtcnn目录中，里面同样也有一张测试后的结果。保存的图片可供以后分析使用。


#### 其它的解决方案使用方法类似，有什么不懂的可以创建issue提问，我尽力解答。

