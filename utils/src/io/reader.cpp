/*============================================================================
 * File Name   : reader.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <fstream>
#include "reader.h"

using namespace std;
using namespace dg;

namespace dg{
Reader::Reader()
{

}

Reader::~Reader()
{

}

vector<uchar> Reader::Read(const char *filename)
{
    streampos file_size;
    ifstream file(filename, ios::binary);
    file.seekg(0, ios::end);
    file_size = file.tellg();
    file.seekg(0, ios::beg);

    std::vector<uchar> file_data(file_size);
    file.read((char *)&file_data[0], file_size);
    return file_data;
}
}