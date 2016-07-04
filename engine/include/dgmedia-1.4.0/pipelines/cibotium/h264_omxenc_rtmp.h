/*============================================================================
 * File Name   : h264_omxenc_rtmp.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月26日 下午1:52:25
 * Description : 
 * ==========================================================================*/
#ifndef H264_OMXENC_RTMP_H_
#define H264_OMXENC_RTMP_H_

#include "pipelines/cibotium/cibotium_pipeline.h"

namespace dgmedia
{

class H264OMXEncRTMP : public CibotiumPipeline
{
public:
	H264OMXEncRTMP(int width, int height, int fps, int framesize,
			PixFmtType pftype, string uri, NeedData data_input);
	virtual ~H264OMXEncRTMP();

	virtual Error SetQualityLevel(int quality_level);
	virtual Error SetBitrate(int bitrate);
	virtual Error SetIFrameInterval(int iframeinterval);

protected:
	virtual void BuildPipeline();
};

} /* namespace dgmedia */

#endif /* H264_OMXENC_RTMP_H_ */
