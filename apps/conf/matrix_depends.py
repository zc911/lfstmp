#!/usr/bin/python
'''
   usage : download matrix data and libs dependences
   author: hongjiangli@deepglint.com
   date:   2016-08-17
'''

import json
import sys
import getopt
import os
import subprocess

def download(name,url,destdir,dest,md5sum,md5check):
    curdir= os.getcwd()
    os.chdir(destdir)
    if not os.path.exists(dest):
        os.makedirs(dest)
    os.chdir(dest)
    subprocess.call("wget --preserve-permissions %s"% url, shell=True) 
    filename = url.split("/")[-1]
    if md5check == "true":
        filemd5 = subprocess.check_output("md5sum %s"%filename,shell=True).split(" ")[0]
        if filemd5 != md5sum:
            print "file %s checkmd5 failed,please check"%filename
            sys.exit()
    if name != filename:
        base = filename
        namelist = name.split(",")
        for item in namelist:
            print item
            print base
            subprocess.call("ln -s %s %s"%(base,item),shell=True)
            base = item
    os.chdir(curdir)    

def json_handle(jsonfile,destdir):
    kv =json.load(file(jsonfile))
    for item in kv.keys():
        item_len = len(kv[item])
        for subitem in range(0,item_len):
            name = kv[item][subitem]['Name']
            url = kv[item][subitem]['Url']
            path = kv[item][subitem]['Path']
            md5sum = kv[item][subitem]['Md5sum']
            md5check = kv[item][subitem]['Md5check']
            #download url to path and checkmd5
            print "begin to download %s" % name 
            download(name,url,destdir,path,md5sum,md5check)
    
def usage():
    print "#########################################################"
    print "example: matrix_depends.py -f data_conf.json -p bin/Debug" 
    print "-f :must specify data/lib json file"
    print "-p :specify where to storage download files, cur_dir DEFAULT"
    print "-h :print usage()"
    print "#########################################################"

if __name__ == '__main__': 
     opts,args = getopt.getopt(sys.argv[1:],"hf:p:")
     FILE=""
     PATH="./" 
     #get file and path
     for op,value in opts:
         if op == "-f":
              FILE = value
         elif op == "-p":
              PATH = value
         elif op == "-h":
              usage()
              sys.exit()
     # read FILE and download urls to PATH
     json_handle(FILE,PATH)
