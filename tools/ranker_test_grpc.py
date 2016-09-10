#!/usr/bin/env python
# -*- coding: utf8 -*-
import base64
import json
import sys
import time
from grpc.beta import implementations

import ranker_pb2

reload(sys)
sys.setdefaultencoding("utf-8")


def start():
    channel = implementations.insecure_channel("192.168.2.21", 7600)
    stub = ranker_pb2.beta_create_SimilarityService_stub(channel)

    grpcReq = ranker_pb2.FeatureRankingRequest()
    grpcReq.ReqId = 1
    grpcReq.Type = 1

    compareImageFile = open("compare.jpg", "rb")
    compareImageContent = base64.b64encode(compareImageFile.read())
    compareImageFile.close()
    grpcReq.Image.Id = "test"
    grpcReq.Image.BinData = compareImageContent

    v_json_file = open("vehicleinfo_validation.json")
    jsonData = json.load(v_json_file)

    id = 1
    for v in jsonData["Vehicles"]:
        feature = v["Features"]
        for i in range(0, 3333):
            candidate = grpcReq.Candidates.add()
            candidate.Id = id;
            candidate.Feature = feature
            id = id + 1
        if id >= 10000:
            break

    # call ranker service
    startTime = time.time()
    resp = stub.GetRankedVector(grpcReq, 100000*100)
    endTime = time.time()

    print "Ranker cost: %d ms with %d candidates" % ((endTime - startTime) * 1000, id - 1)
    # print resp

    # write ranker results
    # respFile = open("resp.json", 'w')
    # respFile.write(resp.content)
    # respFile.close()


if __name__ == "__main__":
    start()
