/*============================================================================
 * File Name   : cibotium_pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月26日 上午10:22:59
 * Description : 
 * ==========================================================================*/
#ifndef CIBOTIUM_PIPELINE_H_
#define CIBOTIUM_PIPELINE_H_

#include <string.h>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <glib.h>

#include "pipelines/pipeline.h"
#include "utils/colorspace.h"

namespace dgmedia
{

typedef void (*NeedData)(unsigned char *data);

class CibotiumPipeline : public Pipeline
{
public:
	CibotiumPipeline(int width, int height, int fps, int framesize,
			PixFmtType pftype, string uri, NeedData data_input);
	virtual ~CibotiumPipeline();

	Error Initialize();
	Error Run();
	Error Stop();
	int GetWidth();
	int GetHeight();
	int GetFramesize();
	PixFmtType GetPixFmtType();
	void RegisteRuntimeErrorReached(
			RuntimeErrorReached on_runtime_error_reached);
	int GetQualityLevel();
	int GetBitrate();
	int GetIFrameInterval();

	virtual Error SetQualityLevel(int quality_level) = 0;
	virtual Error SetBitrate(int bitrate) = 0;
	virtual Error SetIFrameInterval(int iframeinterval) = 0;

protected:
	GstElement *pipeline_;
	GstAppSrc *src_;
	GstBus *bus_;
	GstMessage *msg_;
	GstStateChangeReturn ret_;
	GMainLoop *loop_;
	GstBuffer *buffer_;
	GstClockTime timestamp_;
	NeedData data_input_;

	int width_;
	int height_;
	int fps_;
	int framesize_;
	int quality_level_;
	int bitrate_;
	int iframeinterval_;
	PixFmtType pftype_;
	string uri_;
	RuntimeErrorReached on_runtime_error_reached_;
	string pipeline_str_;

protected:
	static void on_need_data(GstElement *appsrc, guint size, gpointer data);
	static gboolean on_buscall(GstBus *bus, GstMessage *msg, gpointer data);
	virtual Error BuildPipeline() = 0;
};

} /* namespace dgmedia */

#endif /* CIBOTIUM_PIPELINE_H_ */
