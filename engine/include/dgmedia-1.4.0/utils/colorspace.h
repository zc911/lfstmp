/*============================================================================
 * File Name   : colorspace.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月19日 下午3:24:44
 * Description : 
 * ==========================================================================*/
#ifndef COLORSPACE_H_
#define COLORSPACE_H_

#include <string>

#include "utils/pixel_format.h"

using namespace std;

namespace dgmedia
{

class Colorspace
{
public:
	Colorspace(PixFmtType t);
	virtual ~Colorspace();
	int r_mask;
	int g_mask;
	int b_mask;
	int a_mask;
	int bpp;
	int depth;
	string format;
	string mediatype;
	int endianness;
	PixFmtType type;
	string capsstr;
};

} /* namespace dgmedia */

#endif /* COLORSPACE_H_ */
