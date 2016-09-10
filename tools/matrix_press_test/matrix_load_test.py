#!/usr/bin/env python
# -*- coding: utf8 -*-

# author: mikahou
# date  : 28 May., 2016
# Load Test for Matrix 


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

import witness_pb2
import common_pb2
from grpc.beta import implementations



_TIMEOUT_SECONDS = 60
#_TIMEOUT_SECONDS = 20

SERVER_IP = '192.168.5.11'
#SERVER_IP = '192.168.5.11'
SERVER_PORT = 6500
#SERVER_PORT = 6500

SESSION_ID = 0

URI_HEADER = 'http://192.168.2.16:3002/'
MOUNT_PATH = "/home/chenzhen/testimage/testImg-sz/dayfront"
PHOTO_PATH = "/Brand_Data/company/changhai/SrcData/changhai-test/20000"
PHOTO_PATH = "/"
BATCH_NUM = 8

FUNCTIONS = [1,2,3,4,5,6,7,8]  #9-11: face
#FUNCTIONS = [1,2,3,4,5,6,7,8,9,10,11]
SVR_TPYE = 1 # 1:car 2:face 3:all 0:default(face)

## status info
timeBound = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)
#timeBound = (200, 400, 500, 600, 700, 800, 1000, 1500, 2000, 5000)



#channel = implementations.insecure_channel(SERVER_IP, SERVER_PORT)
#stub = witness_pb2.beta_create_WitnessService_stub(channel)

## global configure infomation
conf = dict(
       timeout = 5000,
       threadNum = 2,
       requestID = 0,
       interval = 5,
       onePass = 0,
       verbose = 1,# 0 means closed
       duration = 0, #how long to run, now default: never stop
       uri_type = 2, # 0 means local path, 1 means http path, 2 means encode to base64
       mode = 1, # 0 means single, 1 means batch
)

## init log
logger = logging.getLogger("MyLogger")
os.system("mkdir -p ./log")
log_name = "./log/matrix.log." + str(conf["requestID"])
logging.basicConfig(level=logging.DEBUG,
           format='[%(asctime)s %(name)s %(levelname)s] %(message)s',
           #format='%(asctime)s %(fsilename)s[line:%(lineno)d] %(levelname)s %(message)s',
           datefmt='%Y-%m-%d %H:%M:%S',
           filename=log_name,
           filemode='w')
handler = logging.handlers.RotatingFileHandler(log_name,
            maxBytes = 20971520, backupCount = 5)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.info(
        "[[time.time:%s]]" % str(int(time.time())))
logger.info(
        "[[loadtest start at %s]]" % str(datetime.datetime.now()))
logger.info("Timeout Threshold: %dms", conf["timeout"])
logger.info("threadNum: %d", conf["threadNum"])


## global stat
stat = dict(
    timeDist = [0] * (len(timeBound) + 1),
    timeDistAll = [0] * (len(timeBound) + 1),

    httpCodeDist = [0]*1000,
    httpCodeDistAll = [0]*1000,

    statusCodeDist = [0]*1000,  #analyze matrix return code
    statusCodeDistAll = [0]*1000,

    requestCount = 0,
    requestCountAll = 0,
    requestTime = 0.0,
    requestTimeAll = 0.0,

    httpErrorCount = 0,
    httpErrorCountAll = 0,
    httpErrorTime = 0.0,
    httpErrorTimeAll = 0.0,

    max_qps = 0.0,
    min_qps = 1000000.0,
    max_time = 0.0,
    min_time = 1000000.0,

    timeoutCount = 0,
    timeoutCountAll = 0,
    timeoutTime = 0.0,
    timeoutTimeAll = 0.0,

    errorCount = 0,
    errorCountAll = 0,
    errorTime = 0.0,
    errorTimeAll = 0.0,

)


## query thread
class QueryThread(threading.Thread):
    def __init__(self, root_path):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.root_path = root_path

    def SingleReq(self, query):
        req = witness_pb2.WitnessRequest()
        req_flag = True

        global SESSION_ID
        req.Context.SessionId = 'singleReq_rpc' + str(SESSION_ID)
        SESSION_ID += 1
        #req.Context.SessionId = 'singleReq_test'

        #for i in range(1,12):
        for i in FUNCTIONS:
            req.Context.Functions.append(i)
        req.Context.Type = SVR_TPYE

        logger.info("raw query:%s" %(query))
        req.Image.Data.Id = '0'
        query = query[0]
        if conf['uri_type'] == 0:
            req.Image.Data.URI = 'file:' + query
        elif conf['uri_type'] == 1:
            req.Image.Data.URI =  URI_HEADER + '/' + query.split(MOUNT_PATH)[1]
        elif conf['uri_type'] == 2:
            #tmp_file = '/mnt/ssd/DeepV/server/aws_data/jiadu/20160111/035527_4191e5ee40153674807f622725380c75.jpg'
            fd = open(query, "rb")
            file_encode = base64.b64encode(fd.read())
            fd.close()
            if len(file_encode) < 1000:
                req_flag = False
                logger.error("file size < 1000,not deal it")
                #fd = open(tmp_file, "rb")
                #file_encode = base64.b64encode(fd.read())
                #fd.close()
            req.Image.Data.BinData = file_encode
        return req_flag, req

    def BatchReq(self, batchQuery):
        req = witness_pb2.WitnessBatchRequest()
        req_flag = True

        global SESSION_ID
        req.Context.SessionId = 'batchReq_rpc' + str(SESSION_ID)
        SESSION_ID += 1
        #req.Context.SessionId = 'batchReq_test'
        #for i in range(1,12):
        for i in FUNCTIONS:
            req.Context.Functions.append(i)
        req.Context.Type = SVR_TPYE
    
        cnt = 0
        for query in batchQuery:
            #print "raw query"
            #print query
            logger.info("raw query:%s" %(query))
            image = req.Images.add()
            image.Data.Id = str(cnt)
            #tmp_file = '/mnt/ssd/DeepV/server/aws_data/jiadu/20160111/035527_4191e5ee40153674807f622725380c75.jpg'
            if conf['uri_type'] == 0:
                image.Data.URI = 'file:' + query  
            elif conf['uri_type'] == 1:
                image.Data.URI =  URI_HEADER + '/' + query.split(MOUNT_PATH)[1]
            elif conf['uri_type'] == 2:
                 file_encode = ''
                 fd = open(query, "rb")
                 file_encode = base64.b64encode(fd.read())
                 fd.close()
                 if len(file_encode) < 1000:
                     req_flag = False
                     logger.error("file size < 1000,not deal it")
                     #fd = open(tmp_file, "rb")
                     #file_encode = base64.b64encode(fd.read())
                     #fd.close()
                 image.Data.BinData = file_encode 
            #print 'image.Data.URI'
            #print image.Data.URI
            cnt += 1
        return req_flag, req

    ## request once
    def Search(self, query):
        channel = implementations.insecure_channel(SERVER_IP, SERVER_PORT)
        stub = witness_pb2.beta_create_WitnessService_stub(channel)
 
        begin = 0
        end = 0
        resp = ''

        ## single
        if conf['mode'] == 0:
            code, req = self.SingleReq(query)
            if code == -1:
                return
            begin = time.time()
            resp = stub.Recognize(req, _TIMEOUT_SECONDS)
            end = time.time()
        ## batch
        elif conf['mode'] == 1:
            code, req = self.BatchReq(query)
            if code == -1:
                return
            begin = time.time()
            resp = stub.BatchRecognize(req, 1800)
            #resp = stub.BatchRecognize(req, _TIMEOUT_SECONDS)
            end = time.time()

        #print "#"*50
        #print resp
        #time.sleep(3)

        #request count
        stat["requestCount"] += 1
        #time cost
        elapse = (end - begin) * 1000
        stat["requestTime"] += elapse

        if elapse > conf["timeout"] :
            stat["timeoutCount"] += 1
            stat["timeoutTime"] += elapse

        for (i, bound) in enumerate(timeBound) :
            if elapse < bound :
                stat["timeDist"][i] += 1
                break
        if elapse > timeBound[-1]:
            stat["timeDist"][-1] += 1


        stat["max_time"] = max(elapse, stat["max_time"])
        stat["min_time"] = min(elapse, stat["min_time"])
        #stat["max_qps"] = 1000.0 / stat["min_time"]
        #stat["min_qps"] = 1000.0 / stat["max_time"]

        # matrix return code
        if resp.Context.Status == '':
            return
        returnCode = int(resp.Context.Status)
        #print "matrix return code:%s" %(returnCode)
        stat["statusCodeDist"][returnCode] += 1
        if returnCode != 200 and resp.Context.Message != "":
            #print "Debug###############"
            logger.error(str(resp.Context))

    def batch_run(self, root_path, batch_num):
        batch_list = []
        for sub_dir_name in os.listdir(root_path):
            temp_path = "%s/%s" %(root_path, sub_dir_name)
            #print "temp_path:%s" %(temp_path)
            if re.search("^\.", sub_dir_name) != None:
                #print "not deal it."
                continue
            if os.path.isfile(temp_path):
                if re.search("\.jpg", temp_path) == None:
                    continue
                #print "-------------------"
                #print temp_path
                batch_list.append(temp_path)
                if len(batch_list) >= batch_num:
                    self.Search(batch_list)
                    batch_list = []
            elif os.path.isdir(temp_path):
                #print "in iter..."
                #self.batch_run(temp_path)
                self.batch_run(temp_path, batch_num)
        if len(batch_list) != 0:
            self.Search(batch_list)
            batch_list = []

    def run(self):
        while conf["onePass"] != 2 :
            #try:
                if conf['mode'] == 0:
                    self.batch_run(self.root_path, 1)
                elif conf['mode'] == 1:
                    self.batch_run(self.root_path, BATCH_NUM)
            #except:
            #    pass


class StatThread(threading.Thread) :
    def __init__(self) :
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.cycleCount = 0
        self.logger = logging.getLogger("StatLogger")
        #mika:make log name with requestID
        log_name = "./log/stat.log." + str(conf["requestID"])
        print "log_name:%s" %(log_name)
        handler = logging.handlers.RotatingFileHandler(log_name,
                maxBytes = 20971520, backupCount = 5)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(
                "[[time.time:%s]]" % str(int(time.time())))
        self.logger.info(
                "[[loadtest start at %s]]" % str(datetime.datetime.now()))
        self.logger.info("Timeout Threshold: %dms", conf["timeout"])
        self.logger.info("threadNum: %d", conf["threadNum"])
    
    def SnapShot(self) :
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
            stat["max_qps"] = max(self.requestCount*1.0/conf["interval"], stat["max_qps"])
            stat["min_qps"] = min(self.requestCount*1.0/conf["interval"], stat["min_qps"])

        # reset stats
        stat["timeDist"] = [0] * (len(timeBound) + 1)
        stat["requestCount"] = 0
        stat["timeoutCount"] = 0
        stat["requestTime"] = 0.0
        stat["timeoutTime"] = 0.0
        stat["httpCodeDist"] = [0] * 1000
        stat["statusCodeDist"] = [0] * 1000
        stat["httpErrorCount"] = 0
        stat["HTTPErrorTime"] =0.0
        stat["errorCount"] = 0
        stat["errorTime"] =0.0


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
            ) :
        elapse *= 1.0
        effectiveCount = requestCount  - errorCount - httpErrorCount;
        effectiveTime = requestTime -  errorTime - HTTPErrorTime;
        if elapse == 0:
            avgTimeoutCount = 0
            avgRequestCount = 0
            avgEffectiveCount = 0
            avgErrorCount = 0
            avgHttpErrorCount = 0
        else:
            avgTimeoutCount = timeoutCount / elapse
            avgRequestCount = requestCount / elapse
            avgEffectiveCount = effectiveCount / elapse
            avgErrorCount = errorCount / elapse
            avgHttpErrorCount = httpErrorCount / elapse

        if requestCount == 0:
            avgRequestTime = 0
        else:
            avgRequestTime = requestTime / requestCount

        if effectiveCount == 0:
            avgEffectiveTime = 0
        else:
            avgEffectiveTime = effectiveTime / effectiveCount

        global BATCH_NUM
        throughput = 0
        time_1w = 0
        if conf['mode'] == 0:
            throughput = int(avgEffectiveCount * 86400)
            if avgEffectiveCount != 0:
                time_1w = 10000 / (avgEffectiveCount)
        elif conf['mode'] == 1:
            throughput = int(avgEffectiveCount * BATCH_NUM * 86400)
            if avgEffectiveCount != 0:
                time_1w = 10000 / (avgEffectiveCount * BATCH_NUM)

        statInfo = "### Search time:%d, ErrorCount:%d, HttpErrorCount:%d,\n"\
                   "### timeoutCount:%d, avgTimeoutCount:%.2f/s,\n" \
                   "### avgQPS/avgEffectiveQPS:%.2f/%.2f/s,\n" \
                   "### avgRequestTime/avgEffectiveRequestTime:%.2fms/%.2fms,\n"\
                   "### maxQPS/minQPS:%.2f/%.2f,\n"\
                   "### maxReqestTime/minRequestTime:%.2fms/%.2fms,\n"\
                   "### Throughput per Day:%d,\n"\
                   "### Timecost per 1W::%.2fs,\n"\
                   "### elapse:%ds\n" % (
                        requestCount, errorCount, httpErrorCount, timeoutCount,
                        avgTimeoutCount, avgRequestCount, avgEffectiveCount,
                        avgRequestTime, avgEffectiveTime,
                        stat["max_qps"], stat["min_qps"],
                        stat["max_time"], stat["min_time"],
                        throughput,
                        time_1w,
                        int(elapse))

        timeDistStr = [];
        timeDistStr.append("Time distribution:")
        for i in range(len(timeDist)) :
            if i == 0 :
                timeDistStr.append("  0   - %-3d ms: %d" % (timeBound[i],
                    timeDist[i]))
            elif i == len(timeBound) :
                timeDistStr.append(">  %-3d ms  : %d" % (timeBound[i - 1],
                    timeDist[i]))
            else :
                timeDistStr.append("  %-3d - %-3d ms: %d" % (timeBound[i - 1],
                    timeBound[i], timeDist[i]))

        resultDistStr = [];

        httpCodeDistStr = [];
        httpCodeDistStr.append("Http Code distribution:")
        for i in range(len(httpCodeDist)) :
            if httpCodeDist[i] == 0 :
                continue;
            httpCodeDistStr.append("  %s : %d" % (str(i), httpCodeDist[i]))


        statusCodeDistStr = [];
        statusCodeDistStr.append("DeepV Status Code distribution:")
        for i in range(len(statusCodeDist)) :
            if statusCodeDist[i] == 0 :
                continue;
            statusCodeDistStr.append("  %s : %d" % (str(i), statusCodeDist[i]))

        # change by mika    
        for i in range(
                max(len(timeDistStr), len(resultDistStr), len(httpCodeDistStr), \
                    len(statusCodeDistStr))) :
            str1 = str2 = str3 = str4 = ""
            if i < len(timeDistStr) : str1 = timeDistStr[i]
            if i < len(resultDistStr) : str2 = resultDistStr[i]
            # added by mika
            if i < len(httpCodeDistStr) : str3 = httpCodeDistStr[i]
            if i < len(statusCodeDistStr) : str4 = statusCodeDistStr[i]
            statInfo += "%-25s %-5s %-25s %-25s\n" % (str1, str2, str3, str4)
            #statInfo += "%-25s %-26s %-25s\n" % (str1, str2, str3)

        return statInfo


    def run(self) :
        while conf["onePass"] != 2 :
            time.sleep(conf["interval"])
            self.cycleCount += 1
            self.SnapShot()
            statInfo = "#" * 26 + str(datetime.datetime.now()) + "#" * 26 + "\n"
            if conf["verbose"] :
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
                    conf["interval"] * self.cycleCount,stat["httpCodeDistAll"],
                    stat["errorCountAll"],stat["errorTimeAll"],
                    stat["httpErrorCountAll"],stat["httpErrorTimeAll"],
                    stat["statusCodeDistAll"])


            print statInfo
            self.logger.info(statInfo)
        os.kill(os.getpid(), signal.SIGKILL)

def quitTest(signum, frame) :
    print "loadtest quit"
    sys.exit(0)

#do load test
def LoadTest() :
    testThreads = []
    for i in range(conf["threadNum"]) :
        thrd = QueryThread(MOUNT_PATH + PHOTO_PATH) 
        testThreads.append(thrd)
        thrd.start()

    try:
        statThread = StatThread()
        statThread.start()
    except:
        #mika:for debug
        #print  sys.exc_info()[0],sys.exc_info()[1]
        print "load test error.............."
        sys.stdout = open('log/error.log.' + str(conf["requestID"]), 'a')
        print  sys.exc_info()[0],sys.exc_info()[1]
        sys.stdout = sys.__stdout__
        sys.exit(0)

    if conf["duration"] != 0 :
        signal.signal(signal.SIGALRM, quitTest)
        signal.alarm(conf["duration"] + 1)

    while True :
        time.sleep(36000)

    sys.exit(0)


if __name__ == "__main__":
    LoadTest()
