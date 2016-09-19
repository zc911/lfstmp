/*============================================================================
 * File Name   : reader.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/
#ifndef MATRIX_UTIL_IO_READER_H_
#define MATRIX_UTIL_IO_READER_H_

#include <vector>
#include <opencv2/core/core.hpp>

namespace dg{
class Reader
{
public:
    Reader();
    virtual ~Reader();

    static std::vector<uchar> Read(const char *filename);
}; //end of class Reader
} //end of namespace dg

#endif //MATRIX_UTIL_IO_READER_H_