#ifndef ALG_BASE_DETECTOR_H
#define ALG_BASE_DETECTOR_H
#pragma once

#include "common/common.h"

class CBaseDetector
{
	 	
public:
	CBaseDetector(){};
	virtual ~CBaseDetector() {};

	virtual bool			LoadModel(const char* fn_model)=0; 
 	virtual LIST_RECT&		Detect(BYTE* img_gray, int width, int height, int width_step, float threshold = 0.9, int object_size_min=-1,int object_size_max=-1)=0;
  
	virtual LIST_ELLIPSE&	Detect(std::vector<BYTE*> list_img_gray, int width, int height, int width_step, const LIST_INT& list_angle, float threshold = 0.9, int object_size_min=-1,int object_size_max=-1)=0;
	virtual RECT            GetRectFromEllipse(const ELLIPSE& ellipse, int width, int height, bool bToOriginal = false)=0;
	virtual void			GetPointFromEllipse(const ELLIPSE& ellipse, int width, int height, float& x, float& y, bool bToOriginal = false)=0;

};

#endif