/*============================================================================
 * File Name   : uri_reader.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/
#ifndef MATRIX_UTIL_IO_URI_READER_H_
#define MATRIX_UTIL_IO_URI_READER_H_

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

namespace dg
{
class UriReader
{
public:
    UriReader();
    virtual ~UriReader();

    static int Read(const std::string uri, std::vector<uchar>& buffer);

private:
    static UriReader reader_; //used for global initialize only

//    static size_t write_callback(void* buffer, size_t size, size_t nmemb, void* stream);
}; //end of class Reader
} //end of namespace dg

#endif //MATRIX_UTIL_IO_URI_READER_H_
