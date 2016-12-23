#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
sys.path.append("../../sdk/python")
import json
import argparse
import matrix_client

def start():
    rec_client = matrix_client.MatrixClient("http://192.168.2.21:6501/rec/image", 20, 0)
    rank_client = matrix_client.MatrixClient("http://192.168.2.21:9901/rank", 20, 1)

    input_file = open("input.txt", "r")
    output_file = open("result.csv", "w")
    line = input_file.readline()
    while line:
        if len(line) == 0:
            print "line empty"
            continue
        line_items = line.split(" ")
        if len(line_items) < 4:
            print "line format invalid " + line
            continue
        name = line_items[1]
        path = line_items[2]
        rec_result = json.loads(rec_client.recognize_single(path))
        if rec_result.has_key("Result") and rec_result['Result'].has_key('Pedestrian') and len(rec_result['Result']['Pedestrian']) > 0 and rec_result['Result']['Pedestrian'][0].has_key('Face') and rec_result['Result']['Pedestrian'][0]['Face'].has_key('Features'):
            feature = rec_result['Result']['Pedestrian'][0]['Face']['Features']

            rank_result = json.loads(rank_client.rank_feature(feature))
            if rank_result.has_key('Candidates') and len(rank_result['Candidates']) > 0:
                candidate = rank_result['Candidates'][0]
                c_score = candidate["Score"]
                c_name = candidate["Name"]
                output_line = name + " " + c_name + " " + str(c_score) + "\n"
                output_file.write(output_line)
            else:
                print "There no rank result: " + path
        else:
            print "There is no rec result: " + path
            print json.dumps(rec_result)

        line = input_file.readline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create input file for face db creator')
    parser.add_argument("-i", dest="input_folder", help="the input folder")
    parser.add_argument("-o", dest="output_path", default="", help="the output file path")
    parser.add_argument("-p", dest="prefix", default="", help="the image prefix image path")
    parser.add_argument("-r", dest="recursive", default=False, help="is recursive?")

start()
