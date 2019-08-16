from flask import Flask
from multiprocessing import Process
import re
import os
import time
import sys
import face_recognition_dlib_tensorflow as frdt

app = Flask(__name__)

@app.before_first_request
def before_first_request():
    flask_pid = os.popen("ps -ef |grep flask").readlines()
    f = open("info.txt", 'a')
    pid = re.findall("\d+",flask_pid[0])[0]
    f.writelines(pid+"\n")
    f.close()

@app.route('/start')
def start():
    P = Process(target=frdt.main_process(),args=())
    P.start()
    return "<h1>摄像头已打开</h1>"

@app.route('/finish')
def finish():
    with open("info.txt",'r') as f:
        info_list = f.readlines()
        flask_temp = 0
        for i in range(len(info_list)):
            temp = re.findall('\d+$', info_list[i])
            if temp:
                flask_temp = temp[0]
        print(flask_temp)
        pid_list = os.popen("ps -ef | grep flask").readlines()
        print(pid_list)
        for i in range(len(pid_list)):
            pid_list[i] = pid_list[i].split()[1]
            print(pid_list)
            if str(pid_list[i]) != flask_temp and flask_temp != 0:
                try:
                    os.popen("kill -15 " + str(pid_list[i]))
                except:
                    os.popen("kill -9 " + str(pid_list[i]))
                print("kill " + str(pid_list[i]) + "\n")
    if os.path.exists("info.txt"):
        os.remove("info.txt")
    time.sleep(4)
    sys.exit()

    return "<h1>摄像头已关闭</h1>"