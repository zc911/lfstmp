#!/usr/bin/env python
# -*- coding: utf8 -*-
import base64
import json
import requests
import sys
import time
from grpc.beta import implementations

sys.path.append("matrix_service")
import ranker_pb2
sys.path.append("py-pb-converters")
import pbjson

reload(sys)
sys.setdefaultencoding("utf-8")

# Ranker service address
IP = "192.168.5.11"
PortRestful = 6501
PortGrpc = 6500
RankerServiceRestful = "http://" + IP + ":" + str(PortRestful) + "/rank"
CandidatesDataFile = "feature_modeltype.json"
# Candidates number
CandidatesCount = 10
CandidatesStartIndex = 0
# The compared image path
CompareImage = "compare_1.jpg"
# The compared image features
CompareImageFeature = "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\ni3FGZ1rLphWtU+sA1wDfANIA7ADQAOIAzgD3ANQA7gDVAPcA0ADzAM8APADXADEA1gA9AM4A\nMwDQADkAwgAyAMgAOQC8ADMAvQD4AAQBzABQALoAYABrABwAUwAYAIcAFQB2ABwAWgAUAS4A\nAwHdAIcA4QCDACEAfQBYANQA0gAlANAAHwAyALIA9QDbACwAtgAhAKYAxQDYAEgAVQDGACIA\nvgATAEgACAGfAHgAmwB9AFQAigBXAHgAKwA5APIAsgCmAE8AjgBSAIEARwB3AEwAjAAzANEA\nGAHfAAkB3AAHAecAUADSAEcA0AA/APkAfABLAEoARQBAACoA1AAsAMAAfwD3AH0A9QCJANsA\nYQDaAFcA5gC+ABcBrwD/AEYA3QAyANsAsQDyAMgA2gC7AOkAgQAMAYcAAAF9AP4A6wDfANIA\n5QDQANQA7AC9AK0A9QCUAO8AngDbAJcA4gCgAAwBlwAMAacAAAGOAAAB/QDUAPEAwABAAM0A\ntQCWAI8AlgAsAPgAvwCTAAcBcgA5AB4AYwBAAEAAiAAoAGEAPQDEADkAwgA8AL4AMgC9AOIA\n4QDkANoA3wDeAJcA4gCOAOcATADgAEUA4QBPAN4AQwDeAPcA1ADyANUA9ADPAO4AzAA8ANYA\nOQDTAD8AzQAyAMgAMQDaAFgA1ADeAIYA4ACCACoAgQAgAHwAPwCGAEgACAHfAAgBpgBPACwA\ntgAgAKUAHwBPACsAOQBIAFUAnwB4AJwAfgD5AHsAbACUAFQAigDrAN8AzABOANoAOQCxAAsB\nrgD+AEMAHgA4ABYA2QAyANAAIQBiANoAVgDmAEsASQDyALIA8ACaAH4A9ACCANwAeQDbACoA\n1AAwALgAeQALAYYA/wB8AP4A3wDSANAA1ADqAMAAmQAKAaYA/wCRAAABugDqAMcA2gC2ANoA\nrAD0AJMA7gDTAOQAPwDYAP0A1AD2AMQASgDTAPgAAwEyALIALQADAb4AkgDoAFUAjABSAE0A\n4wBIAN0ATADWAPQAwwDqAL4A9AC4AOwAuwD0AM8A7ADSAPMAyADqAMwA3QDSACwAtAAhAKUA\nngB3AJsAewBIAAcBKQCAAN8ABwH5AHsA8AB3AN0AhADmAE0AywBNAPYA2QDqAN8ApwBPAHcA\n8ACBANwAdwDaAKwABwGsAPQAKQDTACsAtgBwAOoAVgDkAGYA2ABhANoARgBUAEMAVACeAAcB\nlwAGAaUAAAGXAP0AjQAHAXgACgGNAP4AfQD0AI4A5gCnANwAkADjAL8A5AC5AOkAxQDaAK8A\n2gBKANMAPwDMADgAuwBQAOQAPwDYANgA5ADiAOIA0gDjAPwA0wD3ALUA8QCyAPcAAwEwANkA\n2QA1AB4ATwA4AB4APwCFAA==\n"


def start_grpc():
    channel = implementations.insecure_channel(IP, PortGrpc)
    stub = ranker_pb2.beta_create_SimilarityService_stub(channel)

    grpc_req = ranker_pb2.FeatureRankingRequest()
    grpc_req.ReqId = 1
    grpc_req.Type = 1

    compare_image_file = open(CompareImage, "rb")
    compare_image_content = base64.b64encode(compare_image_file.read())
    compare_image_file.close()
    grpc_req.Image.Id = "test"
    grpc_req.Image.BinData = compare_image_content

    v_json_file = open(CandidatesDataFile)
    json_data= json.load(v_json_file)

    id = 1
    start = 0
    for v in json_data["Vehicles"]:
        if start > CandidatesStartIndex:
            feature = v["Features"]
            if feature == "":
                continue
            candidate = grpc_req.Candidates.add()
            candidate.Id = id
            candidate.Feature = feature
            id = id + 1

        start = start + 1
        if id >= CandidatesCount:
            break

    # append the best candidate to the tail
    best_candidate = grpc_req.Candidates.add()
    best_candidate.Id = id
    best_candidate.Feature = CompareImageFeature

    # write the request data into file in case debug
    req_file = open('req.json', 'w')
    post_json_data = pbjson.pb2json(grpc_req)
    json.dump(post_json_data, req_file)
    req_file.close()

    # call ranker service
    start_time = time.time()
    resp = stub.GetRankedVector(grpc_req, 100000 * 100)
    end_time = time.time()

    print "Ranker cost: %d ms with %d candidates" % ((end_time - start_time) * 1000, id - 1)
    resp_json = pbjson.pb2json(resp)

    # write ranker results
    resp_file = open("resp.json", 'w')
    resp_file.write(resp_json)
    resp_file.close()


def start_restful():
    # load the vehicle recognized results file
    v_json_file = open(CandidatesDataFile)
    json_data = json.load(v_json_file)
    post_json_data = {}

    post_json_data["ReqId"] = 1
    post_json_data["Type"] = 1

    # load the compared image and encode as base64
    compare_image_file = open(CompareImage, "rb")
    compare_image_content = base64.b64encode(compare_image_file.read())
    compare_image_file.close()

    image = {}
    image["Id"] = "test"
    image["BinData"] = compare_image_content

    post_json_data["Image"] = image
    post_json_data["Candidates"] = []

    # append candidates to the post data
    id = 1
    index = 0
    for v in json_data["Vehicles"]:
        if index > CandidatesStartIndex:
            feature = v["Features"]
            if feature == "":
                continue
            candidate = {}
            candidate["Id"] = id
            candidate["Brand"] = v["ModelType"]["Brand"] + v["ModelType"]["SubBrand"] + v["Plate"]["PlateText"]
            candidate["Feature"] = feature
            post_json_data["Candidates"].append(candidate)
            id = id + 1
        index = index + 1
        if id >= CandidatesCount:
            break

    # append the best candidate to the tail
    best_candidate = {}
    best_candidate["Id"] = id
    best_candidate["Feature"] = CompareImageFeature
    post_json_data["Candidates"].append(best_candidate)

    # write the post data into file in case debug
    req_file = open('req.json', 'w')
    json.dump(post_json_data, req_file)
    req_file.close()

    # call ranker service
    start_time = time.time()
    post_header = {"Content-type": "application/json"}
    resp = requests.post(RankerServiceRestful, data=json.dumps(post_json_data), headers=post_header)
    end_time = time.time()

    # print the time cost
    print "Ranker cost: %d ms with %d candidates" % ((end_time - start_time) * 1000, id - 1)

    # write ranker results in to file
    resp_file = open("resp.json", 'w')
    resp_file.write(resp.content)
    resp_file.close()


def print_usage():
    print "Param error!"
    print "Usage: python %s restful|grpc" % (sys.argv[0])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        exit()
    arg = sys.argv[1]
    if arg == "restful":
        start_restful()
    elif arg == "grpc":
        start_grpc()
    else:
        print_usage()
