/*============================================================================
 * File Name   : avi_omxdec_h264_pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月24日 下午2:37:37
 * Description : 
 * ==========================================================================*/
#ifndef AVI_OMXDEC_H264_PIPELINE_H_
#define AVI_OMXDEC_H264_PIPELINE_H_

#include "pipelines/persian/mpeg4_omxdec_h264_pipeline.h"

namespace dgmedia
{

class AVIOMXDecH264Pipeline : public Mpeg4OMXDecH264Pipeline
{
public:
	AVIOMXDecH264Pipeline(string uri, PixFmtType pftype, string nanoseconds,
			string protocols, DecodeFinished on_decode_finished);
	virtual ~AVIOMXDecH264Pipeline();

protected:
	virtual Error InitializeDemuxer();
};

} /* namespace dgmedia */

#endif /* AVI_OMXDEC_H264_PIPELINE_H_ */
