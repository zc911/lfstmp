/*============================================================================
 * File Name   : rtsp_ffdec_h264_pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月19日 上午9:56:03
 * Description : 
 * ==========================================================================*/
#ifndef RTSP_FFDEC_H264_PIPELINE_H_
#define RTSP_FFDEC_H264_PIPELINE_H_

#include "pipelines/persian/persian_pipeline.h"

using namespace std;

namespace dgmedia
{

class RTSPFFDecH264Pipeline : public PersianPipeline
{
public:
	RTSPFFDecH264Pipeline(string uri, PixFmtType pftype,
			DecodeFinished on_decode_finished);
	virtual ~RTSPFFDecH264Pipeline();

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

#endif /* RTSP_FFDEC_H264_PIPELINE_H_ */
