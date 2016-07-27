#!/bin/bash
cd lib
tar -cvf engine_libs.tar ./*
scp engine_libs.tar dell@192.168.2.119:/home/dell/release/home/matrix/libs/
rm engine_libs.tar
