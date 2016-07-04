/*============================================================================
 * File Name   : rtsp_omxdec_h264_pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月19日 下午5:56:30
 * Description : 
 * ==========================================================================*/
#ifndef RTSP_OMXDEC_H264_PIPELINE_H_
#define RTSP_OMXDEC_H264_PIPELINE_H_

#include "pipelines/persian/persian_pipeline.h"

namespace dgmedia
{

class RTSPOMXDecH264Pipeline : public PersianPipeline
{
public:
	RTSPOMXDecH264Pipeline(string uri, PixFmtType pftype,
			DecodeFinished on_decode_finished);
	virtual ~RTSPOMXDecH264Pipeline();

protected:
	virtual Error InitializeSrc();
	virtual Error InitializeDemuxer();
	virtual Error InitializeDecoder();
	virtual Error InitializeColorspace();
	virtual Error CreateLink();
	virtual Error CreateSignal();
	static void on_rtspsrcpad_added(GstElement *element, GstPad *pad,
			gpointer data);
};

} /* namespace dgmedia */

#endif /* RTSP_OMXDEC_H264_PIPELINE_H_ */
