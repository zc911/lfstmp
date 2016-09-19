/*============================================================================
 * File Name   : error.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月18日 下午11:37:30
 * Description : 
 * ==========================================================================*/
#ifndef ERROR_H_
#define ERROR_H_

#include <string>

using namespace std;

namespace dgmedia
{

struct Error
{
	int code;
	string err;
};

} /* namespace dgmedia */

#endif /* ERROR_H_ */
