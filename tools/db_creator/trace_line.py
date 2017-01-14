#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse

def start(input_file, output_file):

    line = input_file.readline()
    while line:
        line = line.replace("\n","")
        for i in range(10):
            line = line + input_file.readline().replace("\n","")
        output_file.write(line+"\n")
        line = input_file.readline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trace the line returns from a file')
    parser.add_argument("-i", dest="input_file_path", help="the input file path")
    parser.add_argument("-o", dest="output_file_path", default="", help="the output file path")

    args = parser.parse_args()
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    if input_file_path == '' or output_file_path == '':
        parser.print_help()
        exit(-1)

    input_file = open(input_file_path, "r")
    output_file = open(output_file_path, 'w')
    start(input_file, output_file,)