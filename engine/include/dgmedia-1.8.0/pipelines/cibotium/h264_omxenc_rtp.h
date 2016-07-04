/*============================================================================
 * File Name   : h264_omxenc_rtp.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年3月8日 下午5:13:22
 * Description : 
 * ==========================================================================*/
#ifndef H264_OMXENC_RTP_H_
#define H264_OMXENC_RTP_H_

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <vector>

#include "pipelines/cibotium/h264_omxenc_rtmp.h"

using namespace boost;
using namespace std;

namespace dgmedia
{

class H264OMXEncRTP : public H264OMXEncRTMP
{
public:
	H264OMXEncRTP(int width, int height, int fps, int framesize,
			PixFmtType pftype, string uri, NeedData data_input);
	virtual ~H264OMXEncRTP();

protected:
	virtual Error BuildPipeline();
};

} /* namespace facedect */

#endif /* H264_OMXENC_RTP_H_ */
