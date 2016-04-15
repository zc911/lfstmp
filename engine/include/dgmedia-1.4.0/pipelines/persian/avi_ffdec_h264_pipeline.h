/*============================================================================
 * File Name   : avi_ffdec_h264_pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月24日 下午2:19:28
 * Description : 
 * ==========================================================================*/
#ifndef AVI_FFDEC_H264_PIPELINE_H_
#define AVI_FFDEC_H264_PIPELINE_H_

#include "pipelines/persian/mpeg4_ffdec_h264_pipeline.h"

namespace dgmedia
{

class AVIFFDecH264Pipeline : public Mpeg4FFDecH264Pipeline
{
public:
	AVIFFDecH264Pipeline(string uri, PixFmtType pftype,
			DecodeFinished on_decode_finished);
	virtual ~AVIFFDecH264Pipeline();

protected:
	virtual Error InitializeDemuxer();
};

} /* namespace dgmedia */

#endif /* AVI_FFDEC_H264_PIPELINE_H_ */
