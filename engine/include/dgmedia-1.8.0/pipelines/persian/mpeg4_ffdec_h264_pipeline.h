/*============================================================================
 * File Name   : mpeg4_ffdec_h264_pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月22日 上午10:08:55
 * Description : 
 * ==========================================================================*/
#ifndef MPEG4_FFDEC_H264_PIPELINE_H_
#define MPEG4_FFDEC_H264_PIPELINE_H_

#include "pipelines/persian/persian_pipeline.h"

namespace dgmedia
{

class Mpeg4FFDecH264Pipeline : public PersianPipeline
{
public:
	Mpeg4FFDecH264Pipeline(string uri, PixFmtType pftype, string nanoseconds,
			string protocols, DecodeFinished on_decode_finished);
	virtual ~Mpeg4FFDecH264Pipeline();

protected:
	virtual Error InitializeSrc();
	virtual Error InitializeDemuxer();
	virtual Error InitializeDecoder();
	virtual Error InitializeColorspace();
};

} /* namespace dgmedia */

#endif /* MPEG4_FFDEC_H264_PIPELINE_H_ */
