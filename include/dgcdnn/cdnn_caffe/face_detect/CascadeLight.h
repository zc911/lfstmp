
#ifndef H_CASCADELIGHT_H
#define H_CASCADELIGHT_H
#pragma once

#include "TypeDef.h"
#include "IntegralImageEx.h"
#include <vector>
#include <list>
#include <iostream>
#include <math.h>
using namespace std;

#define USE_FIX_POINT_NUM

#ifdef USE_FIX_POINT_NUM
typedef int MyFloat;
#define FIX_POINT_SHIFT_BIT 20
#define HALF_FIX_POINT_SHIFT_BIT (FIX_POINT_SHIFT_BIT>>1)
inline float myFloat2float(MyFloat v) {return float(v)/(1<<FIX_POINT_SHIFT_BIT);}
inline MyFloat int2myFloat(int i) {return i<<FIX_POINT_SHIFT_BIT;}
inline MyFloat float2myFloat(float f) {return MyFloat(f*(1<<FIX_POINT_SHIFT_BIT));}
inline MyFloat myMul(MyFloat v1, MyFloat v2) { return (int64(v1)*int64(v2))>>FIX_POINT_SHIFT_BIT; }

struct SLookUpTableL : public vector<MyFloat>
{
};

#else //#ifdef USE_FIX_POINT_NUM
typedef float MyFloat;
typedef vector<float> SLookUpTableL;
inline MyFloat float2myFloat(float f) {return f;}
inline MyFloat myMul(MyFloat v1, MyFloat v2) { return v1*v2; }
#endif

struct SIntegralHaarImageL;

struct SHaarFeatureL
{
	TSRect<int> pos_rect, neg_rect;
	int neg_coef;
};



struct SStrongClassifierL
{
	MyFloat nested_scale, nested_offset;
	MyFloat threshold, ori_threshold;

	vector<SHaarFeatureL> feats;
	vector<MyFloat> feat_values;
	vector<int> feat_per_weak;
	vector<SLookUpTableL> par_thres;
	vector<SLookUpTableL> weak_LUTs;

	vector<int> feat_stream;		//including feats, feat_per_weak
	vector<MyFloat> lut_thres_stream;	//include par_thres and weak_LUTs

	bool Compute(SIntegralHaarImageL &img, MyFloat prev_conf, MyFloat &conf);

	void PrepareOffset(int int_img_step);
	bool ComputeByOffset(SIntegralHaarImageL &img, MyFloat prev_conf, MyFloat &conf);
};

struct SCascadeL
{
	int ref_w, ref_h;
    int int_img_step_for_offset;
	vector<SStrongClassifierL> strongs;

    SCascadeL() {ref_w = ref_h = int_img_step_for_offset = -1;}

	bool Compute(SIntegralHaarImageL &img, int num_used_level, MyFloat &conf, int &passed_level);
	bool Load(istream &is);

	void PrepareOffset(int int_img_step);
	bool ComputeByOffset(SIntegralHaarImageL &img, int num_used_level, MyFloat &conf, int &passed_level);
	void ShiftThresholds(float offset);

};


///////////////////////////////////////////////////////////////////////

struct SIntegralHaarImageL
{
	int width, height, step;							        //width, height and step of gradient images
	TCIntegralImage<uint, uchar> int_img;						//integral image of original image
	TCIntegralImageSqr<uint64, uchar> int_img_sqr;				//integral square image of original image: for zero-mean 1-std normalization

	TSRect<int> slide_wnd;										//current slide window: for normalization
	MyFloat	slide_wnd_mean;
	MyFloat slide_wnd_norm_factor;

	//initialize integral image
	bool Init(int w, int h, int step, uchar *pImg, bool keep_old_step)
	{
		width = w;	height = h;	this->step = step;
		int_img.InitIntegralImage(w, h, step, pImg, keep_old_step);
		int_img_sqr.InitIntegralImage(w, h, step, pImg, keep_old_step);
		slide_wnd.l = slide_wnd.t = slide_wnd.r = slide_wnd.b = 0;
		slide_wnd_norm_factor = 0;
		return true;
	}
	//set sliding window and compute standard deviation
	inline bool SetSlideWnd(int left, int top, int right, int bottom, float min_variance)
	{
		if(left<0 || top<0 || right>width || bottom>height) return false;
		slide_wnd.l = left;		slide_wnd.r = right;		slide_wnd.t = top;		slide_wnd.b = bottom;
		int area = (right-left)*(bottom-top);
		uint sum = int_img.Compute(left, top, right, bottom);
		float mean = float(sum)/area;
		uint64 sum_sqr = int_img_sqr.Compute(left, top, right, bottom);
		float var = float(sum_sqr)/area - mean*mean;
		if(var<min_variance) return false;

		slide_wnd_mean = float2myFloat(mean);
		slide_wnd_norm_factor = float2myFloat(1.0f/sqrt(var));
		int_img.SetSlideWndLeftTop(left, top);
		int_img_sqr.SetSlideWndLeftTop(left, top);
		return true;
	}

	inline MyFloat ComputeNormHaarFeat(SHaarFeatureL &feat)
	{
		int l, t;
		l = slide_wnd.l;
		t = slide_wnd.t;
		int pos_v = int_img.Compute(l+feat.pos_rect.l, t+feat.pos_rect.t, l+feat.pos_rect.r, t+feat.pos_rect.b);
		int neg_v = int_img.Compute(l+feat.neg_rect.l, t+feat.neg_rect.t, l+feat.neg_rect.r, t+feat.neg_rect.b);
		return (pos_v-feat.neg_coef*neg_v)*slide_wnd_norm_factor;
		//return (int(int_img.Compute(feat.pos_rect))-feat.neg_coef*int_img.Compute(feat.neg_rect))*slide_wnd_norm_factor;
	}

	inline MyFloat ComputeNormHaarFeat(int *pFeatStream)
	{
		int pos_v = int_img.ComputeBasedOnOffset(pFeatStream[0], pFeatStream[1], pFeatStream[2], pFeatStream[3]);
		int neg_v = int_img.ComputeBasedOnOffset(pFeatStream[4], pFeatStream[5], pFeatStream[6], pFeatStream[7]);
		return (pos_v-pFeatStream[8]*neg_v)*slide_wnd_norm_factor;
	}

};

///////////////////////////////////////////////////////////////////////

void FastBilinearInterpolation(int w, int h, int step, uchar *_img, float scale_ratio, int &new_w, int &new_h, int &new_step, uchar* &_new_img);

struct SDetRespL
{
	TSRect<float> rect;
	int type;
	MyFloat conf;
};

// Detector for scanning grids in 3D (x,y,scale) parameter space
struct SDetectorL
{
	SIntegralHaarImageL int_img;
	int m_img_w, m_img_h;
	int max_img_w, max_img_h, max_img_step;
	uchar *_img_cache[2];

	SDetectorL(int img_w, int img_h, float max_start_scale);
	~SDetectorL();

	int DetectImage(SCascadeL &cascade,									//cascade detector
		const uchar* _img, const int w, const int h, const int step,	//image information (must be gray image)
		const float min_slide_wnd_var,									//minimum variance of a sliding window (simply skip if lower than this threshold
		const int coarse_step_level,									//number of levels in cascade used for coarse scanning 
		const int num_used_level,										//number of levels used for fine scanning (-1 means using all available levels)
		const int x_cstep, const int y_cstep,							//coarse search step in x and y axis
		const int x_fstep, const int y_fstep,							//fine search step in x and y axis
		const float start_scale, const float end_scale, const float scale_step,	 //start and end scale, as well as the scale step
		const float conf_thres,											//confidence threshold to reject inconfident detection responses
		list<SDetRespL> &det_resp,										//[OUTPUT] list of detection responses
		int &scanned_wnd_num);											//[OUTPUT] number of scanned sliding windows 
    
};

// Probe for testing given points on 2D (x,y) parameter surface.
struct SProbeL
{
	SIntegralHaarImageL int_img;
	int m_img_w, m_img_h;
	int max_img_w, max_img_h, max_img_step;

	SProbeL(int max_img_w, int max_img_h, int max_img_step);

    //initialize image to be probed, must be a gray image
    bool InitImage(const uchar* _img, const int w, const int h, const int step);

    //test the sliding window [left, top, left+ref_width, top+ref_height]
    void ProbeAt(const int left, const int top, const float min_wnd_var, SCascadeL &cascade, int &passed_layer_num, float &confidence);      

};


struct SMergeRespL : public SDetRespL
{
	int det_resp_num;
};


int MergeDetResp(list<SDetRespL> &det_resp,				//list of detection responses
				 float overlap_merge_thres,				//overlap threshold to merge detection responses
				 int min_resp_num,						//detection response number threshold to reject weak merged responses
				 float overlap_reject_thres,			//overlap threshold to reject overlapped merged responses
				 list<SMergeRespL> &merge_resp);		//[OUTPUT] merged responses

#endif
