#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# !/usr/bin/env python
# -*- coding: utf8 -*-

# author: mikahou
# date  : 27 Nov, 2015
# Press Test for DeepV POST


import base64
import datetime
import json
import logging
import logging.handlers
import operator
import os
import re
import requests
import signal
import sys
import threading
import time

# MOUNT_PATH = "/home/chenzhen/testimage/testImg-sz/dayfront/"
MOUNT_PATH = "http://192.168.2.21:3002/testImg-sz/"

endpoints = dict(
    single = "http://127.0.0.1:6500/rec/image",
    batch = "http://127.0.0.1:6500/rec/image" + "/batch",
)


BATCH_NUM = 8


FUNCTION_DEF = dict(
    RECFUNC_NONE = 0,
    RECFUNC_VEHICLE = 1,
    RECFUNC_VEHICLE_DETECT = 2,
    RECFUNC_VEHICLE_TRACK = 3,
    RECFUNC_VEHICLE_STYLE = 4,
    RECFUNC_VEHICLE_COLOR = 5,
    RECFUNC_VEHICLE_MARKER = 6,
    RECFUNC_VEHICLE_PLATE = 7,
    RECFUNC_VEHICLE_FEATURE_VECTOR = 8,
    RECFUNC_FACE = 9,
    RECFUNC_FACE_DETECTOR = 10,
    RECFUNC_FACE_FEATURE_VECTOR = 11,
)
FUNCTIONS = [1,2,3,4,5,6,7]
# status info
timeBound = (200, 400, 500, 600, 700, 800, 1000, 1500, 2000, 5000)


# global configure infomation
conf = dict(
    timeout=5000,
    threadNum=1,
    requestID=47,
    interval=1,
    onePass=0,
    verbose=1,  # 0 means closed
    duration=0,  # how long to run, now default: never stop
    encode=0,  # 1 means encode to base64
    recursive=1,
    showstat=1
)

# init log
logger = logging.getLogger("MyLogger")
os.system("mkdir -p ./log")
log_name = "./log/press_error.log." + str(conf["requestID"])
# logging.basicConfig(level=logging.ERROR,
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s %(name)s %(levelname)s] %(message)s',
                    # format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_name,
                    filemode='w')
handler = logging.handlers.RotatingFileHandler(log_name,
                                               maxBytes=20971520, backupCount=5)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.ERROR)
logger.info(
    "[[time.time:%s]]" % str(int(time.time())))
logger.info(
    "[[presstest start at %s]]" % str(datetime.datetime.now()))
logger.info("Timeout Threshold: %dms", conf["timeout"])
logger.info("threadNum: %d", conf["threadNum"])

# global stat
stat = dict(
    timeDist=[0] * (len(timeBound) + 1),
    timeDistAll=[0] * (len(timeBound) + 1),

    httpCodeDist=[0] * 1000,
    httpCodeDistAll=[0] * 1000,

    statusCodeDist=[0] * 1000,  # analyze deepv return code
    statusCodeDistAll=[0] * 1000,

    requestCount=0,
    requestCountAll=0,
    requestTime=0.0,
    requestTimeAll=0.0,

    httpErrorCount=0,
    httpErrorCountAll=0,
    httpErrorTime=0.0,
    httpErrorTimeAll=0.0,

    max_qps=0.0,
    min_qps=1000000.0,
    max_time=0.0,
    min_time=1000000.0,

    timeoutCount=0,
    timeoutCountAll=0,
    timeoutTime=0.0,
    timeoutTimeAll=0.0,

    errorCount=0,
    errorCountAll=0,
    errorTime=0.0,
    errorTimeAll=0.0,

)

SESSION_ID = 0

# query thread
class QueryThread(threading.Thread):
    def __init__(self, root_path):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.root_path = root_path
        self.header = {"Content-type": "application/json"}
        self.functions = FUNCTIONS

    def singleRec(self, query):
        if len(query) == 0:
            return

        post_dict = {"Context": {"SessionId": str(++SESSION_ID), "Functions" : self.functions}}
        post_dict['Image'] = {"Data": query[0]}
        return json.dumps(post_dict)


    def batchRec(self, batchQuery):
        post_dict = {"Context": {"SessionId": str(++SESSION_ID), "Functions" : self.functions}}
        
        batch_image = []
        for image in batchQuery:
            batch_image.append({"Data": image})

      
        post_dict['Images']= batch_image
        return json.dumps(post_dict)

    ## request once
    def Search(self, query, mode = "Single"):

        postdata = ""
        m_url = ""
        
        if mode == "Single":
            postdata = self.singleRec(query)
            m_url = endpoints["single"]
        elif mode == "Batch":
            postdata = self.batchRec(query)
            m_url = endpoints["batch"]
        else:
            print("Invalid search mode: ", mode)

        begin = time.time()
        try:
            resp = requests.post(m_url, data=postdata, headers=self.header)
        except:
            stat["httpErrorCount"] += 1
            stat["httpErrorTime"] += elapse
            return
        
        end = time.time()

        # request count
        stat["requestCount"] += 1
        # time cost
        elapse = (end - begin) * 1000
        stat["requestTime"] += elapse

        # http return code
        resultCode = resp.status_code
        # print "http return code:%s" %(resp.status_code)
        stat["httpCodeDist"][resultCode] += 1

        # http error request
        if int(resultCode) != 200:
            # http error
            stat["httpErrorCount"] += 1
            stat["httpErrorTime"] += elapse

        if elapse > conf["timeout"]:
            stat["timeoutCount"] += 1
            stat["timeoutTime"] += elapse

        for (i, bound) in enumerate(timeBound):
            if elapse < bound:
                stat["timeDist"][i] += 1
                break
        if elapse > timeBound[-1]:
            stat["timeDist"][-1] += 1

        stat["max_time"] = max(elapse, stat["max_time"])
        stat["min_time"] = min(elapse, stat["min_time"])
        



        # deepv return code
        # resp_dict = json.loads(resp.content)
        # returnCode = int(resp_dict["Status"])
        # # print "deepv return code:%s" %(returnCode)
        # stat["statusCodeDist"][returnCode] += 1

        # if returnCode != 200 and resp_dict.has_key("Message"):
        #     # print "Debug###############"
        #     # print resp_dict
        #     # print resp_dict["Message"]
        #     logger.error(resp_dict)



    def run_http_path(self, http_path, batch_num, recursive = True):
        resp = requests.get(http_path)
        if resp.status_code != 200:
            print("Get http path faild: ", http_path)
            return

        items = resp.text.splitlines()
        batch_list = []
        for i in items:
            if i.startswith("<a") != True:
                continue

            link = i[9:i.find("\">")]
            if re.search("^\.", link) != None:
                continue

            link = http_path + link
            if link.endswith(".jpg") == False & recursive :
                self.run_http_path(link, batch_num, recursive)

            if link.endswith(".jpg"):
                if conf["encode"] == 0:

                    batch_list.append({"URI": link})
                elif conf["encode"] == 1:
                    # batch_list.append({"ImageURI":"file:"+temp_path})
                        # print jpg_name
                    fd = open(link, "rb")
                    resp = requests.get(link)
                    if resp.status_code != 200:
                        print("Get http path faild: ", http_path)
                        continue
                    file_encode = base64.b64encode(resp.text)
                    
                    # print file_encode
                    batch_list.append({"BinData": file_encode})

                if len(batch_list) >= batch_num:
                    self.Search(batch_list, "Batch")
                    batch_list = []


    def run_local_path(self, root_path, batch_num):
        batch_list = []
        for sub_dir_name in os.listdir(root_path):
                # for sub_dir_name in os.listdir(MOUNT_PATH):
                temp_path = "%s/%s" % (root_path, sub_dir_name)
                # print "temp_path:%s" %(temp_path)
                if re.search("^\.", sub_dir_name) != None:
                    # print "not deal it."
                    continue
                if os.path.isfile(temp_path):
                    if re.search("\.jpg", temp_path) == None:
                        continue
                    # print "-------------------"
                    # print temp_path
                    if conf["encode"] == 0:
                        batch_list.append({"URI": "file:" + temp_path})
                    elif conf["encode"] == 1:
                        # batch_list.append({"ImageURI":"file:"+temp_path})
                        # print jpg_name
                        fd = open(temp_path, "rb")
                        file_encode = base64.b64encode(fd.read())
                        fd.close()
                        # print file_encode
                        batch_list.append({"BinData": file_encode})
                    if len(batch_list) >= batch_num:
                        self.Search(batch_list, "Batch")
                        batch_list = []
                        # self.Search(temp_path)
                        # break
                elif os.path.isdir(temp_path):
                    # print "in iter..."
                    # self.batch_run(temp_path)
                    self.batch_run(temp_path, batch_num)

        



    def batch_run(self, root_path, batch_num):
        if root_path.startswith("http://"):
            self.run_http_path(root_path, batch_num, conf["recursive"])
        else:
            self.run_local_path(root_path, batch_num, conf["recursive"])
            

    def run(self):
        while conf["onePass"] != 2:
            try:
                self.batch_run(self.root_path, BATCH_NUM)
            except:
                pass


class StatThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.cycleCount = 0
        self.logger = logging.getLogger("StatLogger")
        # mika:make log name with requestID
        log_name = "./log/press_stat.log." + str(conf["requestID"])
        print "log_name:%s" % (log_name)
        handler = logging.handlers.RotatingFileHandler(log_name,
                                                       maxBytes=20971520, backupCount=5)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # self.logger.setLevel(logging.DEBUG)
        self.logger.info(
            "[[time.time:%s]]" % str(int(time.time())))
        self.logger.info(
            "[[presstest start at %s]]" % str(datetime.datetime.now()))
        self.logger.info("Timeout Threshold: %dms", conf["timeout"])
        self.logger.info("threadNum: %d", conf["threadNum"])

    def SnapShot(self):
        # save stats
        self.timeDist = stat["timeDist"][:]
        self.requestCount = stat["requestCount"]
        self.timeoutCount = stat["timeoutCount"]
        self.requestTime = stat["requestTime"]
        self.timeoutTime = stat["timeoutTime"]
        self.httpCodeDist = stat["httpCodeDist"][:]
        self.statusCodeDist = stat["statusCodeDist"][:]
        self.httpErrorCount = stat["httpErrorCount"]
        self.httpErrorTime = stat["httpErrorTime"]
        self.errorCount = stat["errorCount"]
        self.errorTime = stat["errorTime"]

        if self.requestCount != 0:
            stat["max_qps"] = max(self.requestCount * 1.0 / conf["interval"], stat["max_qps"])
            stat["min_qps"] = min(self.requestCount * 1.0 / conf["interval"], stat["min_qps"])
            # stat["max_time"] = max(self.requestTime*1.0/self.requestCount, stat["max_time"])
            # stat["min_time"] = min(self.requestTime*1.0/self.requestCount, stat["min_time"])

        # reset stats
        stat["timeDist"] = [0] * (len(timeBound) + 1)
        stat["requestCount"] = 0
        stat["timeoutCount"] = 0
        stat["requestTime"] = 0.0
        stat["timeoutTime"] = 0.0
        stat["httpCodeDist"] = [0] * 1000
        stat["statusCodeDist"] = [0] * 1000
        stat["httpErrorCount"] = 0
        stat["HTTPErrorTime"] = 0.0
        stat["errorCount"] = 0
        stat["errorTime"] = 0.0

        # count all
        stat["timeDistAll"] = map(operator.add,
                                  stat["timeDistAll"], self.timeDist)
        stat["requestCountAll"] += self.requestCount
        stat["timeoutCountAll"] += self.timeoutCount
        stat["requestTimeAll"] += self.requestTime
        stat["timeoutTimeAll"] += self.timeoutTime
        stat["httpCodeDistAll"] = map(operator.add,
                                      stat["httpCodeDistAll"], self.httpCodeDist)
        stat["statusCodeDistAll"] = map(operator.add,
                                        stat["statusCodeDistAll"], self.statusCodeDist)
        stat["httpErrorCountAll"] += self.httpErrorCount
        stat["httpErrorTimeAll"] += self.httpErrorTime
        stat["errorCountAll"] += self.errorCount
        stat["errorTimeAll"] += self.errorTime

    def StatInfo(self, requestCount, timeoutCount, requestTime,
                 timeoutTime, timeDist, elapse,
                 httpCodeDist, errorCount, errorTime, httpErrorCount, HTTPErrorTime,
                 statusCodeDist
                 ):
        elapse *= 1.0
        effectiveCount = requestCount - errorCount - httpErrorCount;
        effectiveTime = requestTime - errorTime - HTTPErrorTime;
        if elapse == 0:
            avgTimeoutCount = 0
            avgRequestCount = 0
            avgEffectiveCount = 0
            avgErrorCount = 0
            avgHttpErrorCount = 0
            avgEffectiveImages = 0
        else:
            avgTimeoutCount = timeoutCount / elapse
            avgRequestCount = requestCount / elapse
            avgEffectiveCount = effectiveCount / elapse
            avgEffectiveImages = avgEffectiveCount * BATCH_NUM
            avgErrorCount = errorCount / elapse
            avgHttpErrorCount = httpErrorCount / elapse
            throughputPerDay = avgEffectiveCount * BATCH_NUM * 3600 * 24 / 10000

        if requestCount == 0:
            avgRequestTime = 0
        else:
            avgRequestTime = requestTime / requestCount

        if effectiveCount == 0:
            avgEffectiveTime = 0
        else:
            avgEffectiveTime = effectiveTime / effectiveCount

        statInfo = "### Search time:%d, ErrorCount:%d, HttpErrorCount:%d,\n" \
                   "### timeoutCount:%d, avgTimeoutCount:%.2f/s,\n" \
                   "### avgQPS/avgEffectiveQPS:%.2f/%.2f/s,\n" \
                   "### avgEffectiveImages: %.2f \n" \
                   "### avgRequestTime/avgEffectiveRequestTime:%.2fms/%.2fms,\n" \
                   "### maxQPS/minQPS:%.2f/%.2f,\n" \
                   "### maxReqestTime/minRequestTime:%.2fms/%.2fms,\n" \
                   "### throughput/day: %.2f w \n"\
                   "### elapse:%ds\n" % (
                       requestCount, errorCount, httpErrorCount, timeoutCount,
                       avgTimeoutCount, avgRequestCount, avgEffectiveCount, avgEffectiveImages,
                       avgRequestTime, avgEffectiveTime,
                       stat["max_qps"], stat["min_qps"],
                       stat["max_time"], stat["min_time"],
                       throughputPerDay,
                       int(elapse))

        timeDistStr = [];
        timeDistStr.append("Time distribution:")
        for i in range(len(timeDist)):
            if i == 0:
                timeDistStr.append("  0   - %-3d ms: %d" % (timeBound[i],
                                                            timeDist[i]))
            elif i == len(timeBound):
                timeDistStr.append(">  %-3d ms  : %d" % (timeBound[i - 1],
                                                         timeDist[i]))
            else:
                timeDistStr.append("  %-3d - %-3d ms: %d" % (timeBound[i - 1],
                                                             timeBound[i], timeDist[i]))

        resultDistStr = [];

        httpCodeDistStr = [];
        httpCodeDistStr.append("Http Code distribution:")
        for i in range(len(httpCodeDist)):
            if httpCodeDist[i] == 0:
                continue;
            httpCodeDistStr.append("  %s : %d" % (str(i), httpCodeDist[i]))

        statusCodeDistStr = [];
        statusCodeDistStr.append("DeepV Status Code distribution:")
        for i in range(len(statusCodeDist)):
            if statusCodeDist[i] == 0:
                continue;
            statusCodeDistStr.append("  %s : %d" % (str(i), statusCodeDist[i]))

        # change by mika
        for i in range(
                max(len(timeDistStr), len(resultDistStr), len(httpCodeDistStr), \
                    len(statusCodeDistStr))):
            str1 = str2 = str3 = str4 = ""
            if i < len(timeDistStr): str1 = timeDistStr[i]
            if i < len(resultDistStr): str2 = resultDistStr[i]
            # added by mika
            if i < len(httpCodeDistStr): str3 = httpCodeDistStr[i]
            if i < len(statusCodeDistStr): str4 = statusCodeDistStr[i]
            statInfo += "%-25s %-5s %-25s %-25s\n" % (str1, str2, str3, str4)
            # statInfo += "%-25s %-26s %-25s\n" % (str1, str2, str3)

        return statInfo

    def run(self):
        while conf["onePass"] != 2:
            time.sleep(conf["interval"])
            self.cycleCount += 1
            self.SnapShot()
            statInfo = "#" * 26 + str(datetime.datetime.now()) + "#" * 26 + "\n"
            if conf["verbose"]:
                statInfo += "verbose:\n"
                statInfo += self.StatInfo(self.requestCount,
                                          self.timeoutCount, self.requestTime,
                                          self.timeoutTime, self.timeDist,
                                          conf["interval"], self.httpCodeDist,
                                          self.errorCount, self.errorTime, self.httpErrorCount,
                                          self.httpErrorTime, self.statusCodeDist)
                statInfo += "=-" * 38 + "\n"

            statInfo += self.StatInfo(
                stat["requestCountAll"],
                stat["timeoutCountAll"], stat["requestTimeAll"],
                stat["timeoutTimeAll"],
                stat["timeDistAll"],
                conf["interval"] * self.cycleCount, stat["httpCodeDistAll"],
                stat["errorCountAll"], stat["errorTimeAll"],
                stat["httpErrorCountAll"], stat["httpErrorTimeAll"],
                stat["statusCodeDistAll"])

            print statInfo
            self.logger.info(statInfo)
        os.kill(os.getpid(), signal.SIGKILL)


def quitTest(signum, frame):
    print "presstest quit"
    sys.exit(0)


# do press test
def PressTest():
    # set requests module log level
    logger1 = logging.getLogger("urllib3.connectionpool")
    logger1.setLevel(logging.ERROR)
    # logger2 = logging.getLogger("StatLogger")
    # logger2.setLevel(logging.ERROR)


    testThreads = []
    for i in range(conf["threadNum"]):
        thrd = QueryThread(MOUNT_PATH)
        testThreads.append(thrd)
        thrd.start()

    if conf["showstat"] == 1:
        try:
            statThread = StatThread()
            statThread.start()
        except:
            # mika:for debug
            # print  sys.exc_info()[0],sys.exc_info()[1]
            print "press error.............."
            sys.stdout = open('error.log.' + str(conf["requestID"]), 'a')
            print  sys.exc_info()[0], sys.exc_info()[1]
            sys.stdout = sys.__stdout__
            sys.exit(0)
    
    if conf["duration"] != 0:
        signal.signal(signal.SIGALRM, quitTest)
        signal.alarm(conf["duration"] + 1)

    while True:
        time.sleep(36000)

    sys.exit(0)


if __name__ == "__main__":
    # set requests module log level
    # logger = logging.getLogger("connectionpool")
    # logger.setLevel(logging.WARNING)

    # p =QueryThread(MOUNT_PATH)
    # p.Search("{20141114-1633-4772-5502-CEBDA6E53E44}.jpg")
    # p.run()
    # logger = logging.getLogger("requests.packages.urllib3.connectionpool")
    # logger.setLevel(logging.ERROR)
    PressTest()
