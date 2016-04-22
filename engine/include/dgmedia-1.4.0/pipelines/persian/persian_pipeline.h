/*============================================================================
 * File Name   : persian_pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月18日 下午8:30:13
 * Description : 
 * ==========================================================================*/
#ifndef PERSIAN_PIPELINE_H_
#define PERSIAN_PIPELINE_H_

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <glib.h>

#include "pipelines/pipeline.h"
#include "models/frameset.h"
#include "utils/colorspace.h"

namespace dgmedia
{

typedef void (*DecodeFinished)(unsigned char *data, int size, Frameset info);

typedef void (*EOSReached)();

class PersianPipeline : public Pipeline
{
public:
	PersianPipeline(string uri, PixFmtType pftype, DecodeFinished on_decode_finished);
	virtual ~PersianPipeline();

	Error Initialize();
	Error Run();
	Error Stop();
	Error SetResizeResolution(int width, int height);
	int GetResizeResolutionWidth();
	int GetResizeResolutionHeight();
	int GetSrcResolutionWidth();
	int GetSrcResolutionHeight();
	PixFmtType GetPixFmtType();
	Error SetRepeat(bool repeat);
	bool GetRepeat();
	void RegisteRuntimeErrorReached(RuntimeErrorReached on_runtime_error_reached);
	void RegisteEOSReached(EOSReached on_eos_reached);

protected:
	GstElement *pipeline_;
	GstElement *src_;
	GstElement *demuxer_;
	GstElement *decoder_;
	GstElement *colorspace_;
	GstElement *sink_;
	GstBus *bus_;
	GstMessage *msg_;
	GstStateChangeReturn ret_;
	GMainLoop *loop_;
	GstCaps *caps_;

	int resize_width_;
	int resize_height_;
	int src_width_;
	int src_height_;
	bool repeat_;
	PixFmtType pftype_;
	string uri_;
	DecodeFinished on_decode_finished_;
	EOSReached on_eos_reached_;
	RuntimeErrorReached on_runtime_error_reached_;
	int64_t frame_index_;

protected:
	virtual Error InitializeSrc() = 0;
	virtual Error InitializeDemuxer() = 0;
	virtual Error InitializeDecoder() = 0;
	virtual Error InitializeColorspace() = 0;
	virtual Error CreateLink() = 0;
	virtual Error CreateSignal() = 0;
	static gboolean on_buscall(GstBus *bus, GstMessage *msg, gpointer data);
	static void on_sinkpad_added(GstElement *sink, PersianPipeline *_this);

private:
	Frameset GetFrameset();
};

} /* namespace msrc */

#endif /* PERSIAN_PIPELINE_H_ */
