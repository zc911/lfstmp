#!/usr/bin/env python
# -*- coding: utf8 -*-
import base64
import json
import requests
import sys
import time

reload(sys)
sys.setdefaultencoding("utf-8")


def start():
    v_json_file = open("v_results.json")
    jsonData = json.load(v_json_file)
    postJsonData = {}

    postJsonData["ReqId"] = 1
    postJsonData["Type"] = 1

    # compareFeature = jsonData["Vehicles"][1]["Features"]
    compareImageFile = open("compare.jpg", "rb")
    compareImageContent = base64.b64encode(compareImageFile.read())
    compareImageFile.close()

    image = {}
    image["Id"] = "test"
    image["BinData"] = compareImageContent

    postJsonData["Image"] = image
    postJsonData["Candidates"] = []

    id = 1
    for v in jsonData["Vehicles"]:
        feature = v["Features"]
        candidate = {}
        candidate["Id"] = id
        candidate["Feature"] = feature
        postJsonData["Candidates"].append(candidate)
        id = id + 1

    # reqFile = open('req.json', 'w')
    # json.dump(postJsonData, reqFile)
    # reqFile.close()

    # call ranker service
    startTime = time.time()
    postHeader = {"Content-type": "application/json"}
    resp = requests.post("http://192.168.2.21:7601/rank", data=json.dumps(postJsonData), headers=postHeader)
    endTime = time.time()

    print "Ranker cost: %d ms with %d candidates" % ((endTime - startTime) * 1000, id - 1)
    print resp.content

    # write ranker results
    respFile = open("resp.json", 'w')
    respFile.write(resp.content)
    respFile.close()


if __name__ == "__main__":
    start()
