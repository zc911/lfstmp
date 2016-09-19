/*============================================================================
 * File Name   : video_frame_info.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月18日 下午10:01:54
 * Description : 
 * ==========================================================================*/
#ifndef VIDEO_FRAME_INFO_H_
#define VIDEO_FRAME_INFO_H_

#include <stdint.h>

#include "utils/pixel_format.h"

namespace dgmedia
{

struct Frameset
{
	int resize_width;
	int resize_height;
	int src_width;
	int src_height;
	int64_t index;
	PixFmtType pftype;
	PipelineStatus status;
};

} /* namespace dgmedia */

#endif /* VIDEO_FRAME_INFO_H_ */
