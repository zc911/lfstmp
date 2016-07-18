#!/bin/bash
cd lib
tar -cvf apps_libs.tar ./*
scp apps_libs.tar dell@192.168.2.119:/home/dell/release/home/matrix/libs/
rm apps_libs.tar
