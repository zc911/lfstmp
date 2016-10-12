#ifndef ALG_FACE_DETECTOR_EX_H
#define ALG_FACE_DETECTOR_EX_H

#include "common/common.h"
#include "CascadeLight.h"

#include "AlgDetFace.h"

class CFaceDetectorEx: public CBaseDetector
{
	SCascadeL*  m_cascade;
	SDetectorL* m_detector;
	
public:
	CFaceDetectorEx(const char* fn_model);
	virtual ~CFaceDetectorEx();

	bool			LoadModel(const char* fn_model); 
 	LIST_RECT&		Detect(BYTE* img_gray, int width, int height, int width_step, float threshold = 0.9, int face_size_min=-1,int face_size_max=-1);
	
	LIST_ELLIPSE&	Detect(std::vector<BYTE*> list_img_gray, int width, int height, int width_step, const LIST_INT& list_angle, float threshold = 0.9, int face_size_min=-1,int face_size_max=-1);
	RECT            GetRectFromEllipse(const ELLIPSE& ellipse, int width, int height, bool bToOriginal = false);
	void			GetPointFromEllipse(const ELLIPSE& ellipse, int width, int height, float& x, float& y, bool bToOriginal = false);

protected:

	void Clear();

protected:
	float        m_threshold;
	float        m_scale_step;
	LIST_RECT    m_list_rect;
	LIST_ELLIPSE m_list_ellipse;

};

#endif