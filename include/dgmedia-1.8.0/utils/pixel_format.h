/*============================================================================
 * File Name   : pixel_format.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月18日 下午10:06:59
 * Description : 
 * ==========================================================================*/
#ifndef PIXEL_FORMAT_H_
#define PIXEL_FORMAT_H_

namespace dgmedia
{

typedef enum
{
	//RGB
	RGB24 = 0,     				// packed RGB 8:8:8, 24bpp, RGBRGB...
	BGR24 = 1,     				// packed RGB 8:8:8, 24bpp, BGRBGR...
	ARGB = 2,     				// packed ARGB 8:8:8:8, 32bpp, ARGBARGB...
	RGBA = 3,      				// packed RGBA 8:8:8:8, 32bpp, RGBARGBA...
	ABGR = 4,      				// packed ABGR 8:8:8:8, 32bpp, ABGRABGR...
	BGRA = 5,      				// packed BGRA 8:8:8:8, 32bpp, BGRABGRA...
	RGBX = 6,					// packed RGBX 8:8:8:8, 32bpp, RGBXRGBX...
	XRGB = 7,					// packed XRGB 8:8:8:8, 32bpp, XRGBXRGB...
	BGRX = 8,					// packed BGRX 8:8:8:8, 32bpp, BGRXBGRX...
	XBGR = 9,					// packed XBGR 8:8:8:8, 32bpp, XBGRXBGR...

	//GREY
	GREY8 = 10,     				//        Y        ,  8bpp
	GREY16BE = 11,  				//        Y        , 16bpp, big-endian
	GREY16LE = 12,  				//        Y        , 16bpp, little-endian

	//YUV
	I420 = 13, 	// planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
	NV12 = 14, // planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
	NV21 = 15,      				// as above, but U and V bytes are swapped
	YV12 = 16,
	YUY2 = 17,
	Y444 = 18,
	GREY = 19
} PixFmtType;

} /* namespace dgmedia */

#endif /* PIXEL_FORMAT_H_ */
