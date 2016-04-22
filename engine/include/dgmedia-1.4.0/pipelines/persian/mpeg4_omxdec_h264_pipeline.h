/*============================================================================
 * File Name   : mpeg4_omxdec_h264_pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月22日 上午11:04:17
 * Description : 
 * ==========================================================================*/
#ifndef MPEG4_OMXDEC_H264_PIPELINE_H_
#define MPEG4_OMXDEC_H264_PIPELINE_H_

#include "pipelines/persian/persian_pipeline.h"

namespace dgmedia
{

class Mpeg4OMXDecH264Pipeline : public PersianPipeline
{
public:
	Mpeg4OMXDecH264Pipeline(string uri, PixFmtType pftype,
			DecodeFinished on_decode_finished);
	virtual ~Mpeg4OMXDecH264Pipeline();

protected:
	virtual Error InitializeSrc();
	virtual Error InitializeDemuxer();
	virtual Error InitializeDecoder();
	virtual Error InitializeColorspace();
	virtual Error CreateLink();
	virtual Error CreateSignal();
	static void on_qtdemuxpad_added(GstElement *element, GstPad *pad,
			gpointer data);
};

} /* namespace dgmedia */

#endif /* MPEG4_OMXDEC_H264_PIPELINE_H_ */
