#!/usr/bin/env python
# -*- coding: utf8 -*-
import argparse
import commands
import os
import shutil

ignore_file_ext = [".json", ".cfg", ".cdnn", ".mean", ".a", ".b", ".config", ".c"]


def encrypt(encrypt_cmd, input_file_path, output_file_path):
    print "Encrypt file " + input_file_path + " to " + output_file_path

    status, returnmsg = commands.getstatusoutput('./%s -i %s -o %s' % (encrypt_cmd, input_file_path, output_file_path))
    if status != 0:
        print "Encrypt model %d failed, error msg:%s" % (input_file_path, returnmsg)
    else:
        print "Encrypt file %s to %s successful." % (input_file_path, output_file_path)


def start(input_folder_path, output_folder_path, encrypt_cmd, recursive=True):
    for file_name in os.listdir(input_folder_path):
        full_path = os.path.join(input_folder_path, file_name)

        if os.path.isfile(full_path):
            ext_name = os.path.splitext(file_name)[1]
            output_file_path = os.path.join(output_folder_path, file_name)
            output_folder = output_file_path[:output_file_path.rindex('/')]

            if not os.path.exists(output_folder):
                print "Create folder: " + output_folder
                os.makedirs(output_folder)

            # just copy file without encryption if in ignore list
            if ext_name in ignore_file_ext:
                print "Copy file %s to %s without encryption" % (full_path, output_file_path)
                shutil.copyfile(full_path, output_file_path)
                continue

            encrypt(encrypt_cmd, full_path, output_file_path)
        elif recursive:
            start(full_path, output_folder_path + "/" + file_name, encrypt_cmd, recursive)
        else:
            print "WARNNING: The folder %s contains sub folders but the param resursive is off." % full_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create input file for face db creator')
    parser.add_argument("-i", dest="input_folder", help="the input folder")
    parser.add_argument("-o", dest="output_path", default="encryptModel", help="the output file path")
    parser.add_argument("-p", dest="prefix", default="", help="the image prefix image path")
    parser.add_argument("-r", dest="recursive", default=True, help="is recursive?")
    parser.add_argument("-c", dest="encrypt_cmd", default="model_encrypt", help="the command to encrypt the model")
    parser.add_argument("-m", dest="model_folder_name", default="model",
                        help="the folder name contains unencryption model")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_path = args.output_path
    prefix = args.prefix
    recursive = args.recursive
    encrypt_cmd = args.encrypt_cmd
    model_folder_name = args.model_folder_name

    if input_folder == '':
        parser.print_help()
        exit(-1)

    # output_file = open(output_path, 'w')
    start(input_folder, input_folder.replace(model_folder_name, output_path), encrypt_cmd, recursive)
