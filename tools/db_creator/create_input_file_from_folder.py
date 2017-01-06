#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse
import json
import os.path
import string

id = 0

def start(folder_path, output_file, prefix="", recursive=True):
    global id
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if string.find(full_path, " ") > 0:
            print "Error! find space in the folder: " + full_path
            return
        if os.path.isfile(full_path):
            name = os.path.splitext(file_name)[0]
            path = prefix + "/" + file_name
            attrs = {}
            line = str(id) + " " + str(name) + " " + path + " " + json.dumps(attrs) + "\n"
            output_file.write(line)
            id = id + 1
        else:
            if recursive:
                start(full_path, output_file, prefix+'/'+file_name, recursive)
            else:
                print "not a regular file:" + full_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create input file for face db creator')
    parser.add_argument("-i", dest="input_folder", help="the input folder")
    parser.add_argument("-o", dest="output_path", default="", help="the output file path")
    parser.add_argument("-p", dest="prefix", default="", help="the image prefix image path")
    parser.add_argument("-r", dest="recursive", default=False, help="is recursive?")


    args = parser.parse_args()
    input_folder = args.input_folder
    output_path = args.output_path
    prefix = args.prefix
    recursive = args.recursive

    if input_folder == '' or output_path == '':
        parser.print_help()
        exit(-1)

    output_file = open(output_path, 'w')
    start(input_folder, output_file, prefix, recursive)
