
#ifndef H_INTEGRALIMAGEEX_H
#define H_INTEGRALIMAGEEX_H
#pragma once

#include "TypeDef.h"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

template <class T, class baseT>
class TCIntegralImage
{
public:
	int m_iWidth, m_iHeight, m_iStep;
	int m_iAllocatedMemory;
	T* m_pIImage;
	T* m_pSlideWndLeftTop;
public:
	TCIntegralImage();
	~TCIntegralImage();
	
	T* GetIImage() const {return m_pIImage;}

	bool InitIntegralImage(int w, int h, int step, baseT* pImage=NULL, bool keep_old_step=false);								//compute integral image w.r.t a single image
	
	bool InitIntegralImageCov(int w, int h, int step, baseT* pImageX=NULL, baseT* pImageY=NULL, bool keep_old_step=false);		//compute integral image w.r.t two correlated images
 	
 	inline T Compute(int left, int top, int right, int bottom)
	{
		return m_pIImage[top*m_iStep+left] - m_pIImage[top*m_iStep+right] - m_pIImage[bottom*m_iStep+left] + m_pIImage[bottom*m_iStep+right];
	}
	inline T Compute(TSRect<int> &rect)
	{
		return m_pIImage[rect.t*m_iStep+rect.l] - m_pIImage[rect.t*m_iStep+rect.r] - m_pIImage[rect.b*m_iStep+rect.l] + m_pIImage[rect.b*m_iStep+rect.r];
	}

	//for fast computation
	void SetSlideWndLeftTop(int left, int top) { m_pSlideWndLeftTop = m_pIImage + top*m_iStep + left;}
	
	inline T ComputeBasedOnOffset(int lt_offset, int rt_offset, int lb_offset, int rb_offset)
	{
		return m_pSlideWndLeftTop[lt_offset] - m_pSlideWndLeftTop[rt_offset] - m_pSlideWndLeftTop[lb_offset] + m_pSlideWndLeftTop[rb_offset];
	}
};

template <class T, class baseT>
TCIntegralImage<T, baseT>::TCIntegralImage()
{
	m_pIImage = NULL;
	m_iWidth = m_iHeight = m_iStep = 0;
	m_iAllocatedMemory = 0;
}

template <class T, class baseT>
TCIntegralImage<T, baseT>::~TCIntegralImage()
{
	if(m_pIImage) delete []m_pIImage;
}

template <class T, class baseT>
bool TCIntegralImage<T, baseT>::InitIntegralImage(int w, int h, int step, baseT* pImage, bool keep_old_step)
{
	int new_step = (((w+1)*sizeof(T)+3)>>2<<2)/sizeof(T);
	if((h+1)*new_step>m_iAllocatedMemory){
		//reallocate memory
		if(m_pIImage) delete []m_pIImage;
		m_iWidth = w;	m_iHeight = h;	m_iStep = new_step;
		m_pIImage = new T[(m_iHeight+1)*m_iStep];
		if(m_pIImage==NULL) return false;
		m_iAllocatedMemory = (m_iHeight+1)*m_iStep;
	}else{
		//keep the memory
		m_iWidth = w;	m_iHeight = h;
		if(keep_old_step){
			if(m_iStep<new_step)	//cannot keep the old step
				return false;
		}else{
			m_iStep = new_step;
		}
	}

	if(pImage==NULL) return true;

	memset(m_pIImage, 0, sizeof(T)*m_iStep);
	for(int y=1; y<=m_iHeight; y++){
		m_pIImage[y*m_iStep] = 0;
		T line_sum = 0;
		T* pII = m_pIImage + y*m_iStep + 1;
		T* pIIEnd = pII + m_iWidth;
		T* pIIPrevLine = m_pIImage + (y-1)*m_iStep + 1;
		baseT* pI = pImage+(y-1)*step;
		while(pII!=pIIEnd){
			line_sum += *pI++;
			*pII++ = *(pIIPrevLine++) + line_sum;
		}
	}

	return true;
}

template <class T, class baseT>
bool TCIntegralImage<T, baseT>::InitIntegralImageCov(int w, int h, int step, baseT* pImageX, baseT* pImageY, bool keep_old_step)
{
	int new_step = (((w+1)*sizeof(T)+3)>>2<<2)/sizeof(T);
	if((h+1)*new_step>m_iAllocatedMemory){
		//reallocate memory
		if(m_pIImage) delete []m_pIImage;
		m_iWidth = w;	m_iHeight = h;	m_iStep = new_step;
		m_pIImage = new T[(m_iHeight+1)*m_iStep];
		if(m_pIImage==NULL) return false;
		m_iAllocatedMemory = (m_iHeight+1)*m_iStep;
	}else{
		//keep the memory
		m_iWidth = w;	m_iHeight = h;
		if(keep_old_step){
			if(m_iStep<new_step)	//cannot keep the old step
				return false;
		}else{
			m_iStep = new_step;
		}
	}

	if(pImageX==NULL || pImageY==NULL) return true;

	memset(m_pIImage, 0, sizeof(T)*m_iStep);
	for(int y=1; y<=m_iHeight; y++){
		m_pIImage[y*m_iStep] = 0;
		T line_sum = 0;
		T* pII = m_pIImage + y*m_iStep + 1;
		T* pIIEnd = pII + m_iWidth;
		T* pIIPrevLine = m_pIImage + (y-1)*m_iStep + 1;
		baseT* pX = pImageX+(y-1)*step;
		baseT* pY = pImageY+(y-1)*step;
		while(pII!=pIIEnd){
			line_sum += (*pX++)*(*pY++);
			*pII++ = *(pIIPrevLine++) + line_sum;
		}
	}

	return true;
}

////////////////////////////////////////////////////////////////////

template <class T, class baseT>
class TCIntegralImageSqr  
{
public:
	int m_iWidth, m_iHeight, m_iStep;
	int m_iAllocatedMemory;
	T* m_pIImage;
	T* m_pSlideWndLeftTop;

public:
	TCIntegralImageSqr();
	~TCIntegralImageSqr();
	
	T* GetIImage() const {return m_pIImage;}

	bool InitIntegralImage(int w, int h, int step, baseT* pImage=NULL, bool keep_old_step=false);
 	
	inline T Compute(int left, int top, int right, int bottom)
	{
		return m_pIImage[top*m_iStep+left] - m_pIImage[top*m_iStep+right] - m_pIImage[bottom*m_iStep+left] + m_pIImage[bottom*m_iStep+right];
	}
	inline T Compute(TSRect<int> &rect)
	{
		return m_pIImage[rect.t*m_iStep+rect.l] - m_pIImage[rect.t*m_iStep+rect.r] - m_pIImage[rect.b*m_iStep+rect.l] + m_pIImage[rect.b*m_iStep+rect.r];
	}

	void SetSlideWndLeftTop(int left, int top) { m_pSlideWndLeftTop = m_pIImage + top*m_iStep + left;}

	inline T ComputeBasedOnOffset(int lt_offset, int rt_offset, int lb_offset, int rb_offset)
	{
		return m_pSlideWndLeftTop[lt_offset] - m_pSlideWndLeftTop[rt_offset] - m_pSlideWndLeftTop[lb_offset] + m_pSlideWndLeftTop[rb_offset];
	}
};

template <class T, class baseT>
TCIntegralImageSqr<T, baseT>::TCIntegralImageSqr()
{
	m_pIImage = NULL;
	m_iWidth = m_iHeight = m_iStep = 0;
	m_iAllocatedMemory = 0;
}

template <class T, class baseT>
TCIntegralImageSqr<T, baseT>::~TCIntegralImageSqr()
{
	if(m_pIImage) delete []m_pIImage;
}

 

template <class T, class baseT>
bool TCIntegralImageSqr<T, baseT>::InitIntegralImage(int w, int h, int step, baseT* pImage, bool keep_old_step)
{
	int new_step = (((w+1)*sizeof(T)+3)>>2<<2)/sizeof(T);
	if((h+1)*new_step>m_iAllocatedMemory){
		//reallocate memory
		if(m_pIImage) delete []m_pIImage;
		m_iWidth = w;	m_iHeight = h;	m_iStep = (m_iWidth+1+3)>>2<<2;
		m_pIImage = new T[(m_iHeight+1)*m_iStep];
		if(m_pIImage==NULL) return false;
		m_iAllocatedMemory = (m_iHeight+1)*m_iStep;
	}else{
		//keep the memory
		m_iWidth = w;	m_iHeight = h;
		if(keep_old_step){
			if(m_iStep<new_step)	//cannot keep the old step
				return false;
		}else{
			m_iStep = (m_iWidth+1+3)>>2<<2;
		}
	}

	if(pImage==NULL) return true;

	memset(m_pIImage, 0, sizeof(T)*m_iStep);
	for(int y=1; y<=m_iHeight; y++){
		m_pIImage[y*m_iStep] = 0;
		T line_sum = 0;
		T* pII = m_pIImage + y*m_iStep + 1;
		T* pIIEnd = pII + m_iWidth;
		T* pIIPrevLine = m_pIImage + (y-1)*m_iStep + 1;
		baseT* pI = pImage+(y-1)*step;
		while(pII!=pIIEnd){
			line_sum += (*pI)*(*pI++);
			*pII++ = *(pIIPrevLine++) + line_sum;
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////
//				Multi Channel Integral Image							//
//////////////////////////////////////////////////////////////////////////

template <class T, class baseT>
class TCIntegralImageMC
{
public:
	int m_iWidth, m_iHeight, m_iStep, m_iChannel;
	int m_iAllocatedMemory;
	T* m_pIImage;
	T* m_pSlideWndLeftTop;
public:
	TCIntegralImageMC();
	~TCIntegralImageMC();

	bool InitIntegralImage(int w, int h, int step, int channel, baseT* pImage, bool keep_old_step);				//compute integral image w.r.t a single image

	inline void Compute(int left, int top, int right, int bottom, T* rst) const;	
};

template <class T, class baseT>
TCIntegralImageMC<T, baseT>::TCIntegralImageMC()
{
	m_pIImage = NULL;
	m_iWidth = m_iHeight = m_iStep = m_iChannel = 0;
	m_iAllocatedMemory = 0;
}

template <class T, class baseT>
TCIntegralImageMC<T, baseT>::~TCIntegralImageMC()
{
	if(m_pIImage) delete []m_pIImage;
}

template <class T, class baseT>
bool TCIntegralImageMC<T, baseT>::InitIntegralImage(int w, int h, int step, int channel, baseT* pImage, bool keep_old_step)
{
	int new_step = (((w+1)*channel*sizeof(T)+3)>>2<<2)/sizeof(T);
	if((h+1)*new_step>m_iAllocatedMemory){
		//reallocate memory
		if(m_pIImage) delete []m_pIImage;
		m_iWidth = w;	m_iHeight = h;	m_iStep = new_step; m_iChannel = channel;
		m_pIImage = new T[(m_iHeight+1)*m_iStep];
		if(m_pIImage==NULL) return false;
		m_iAllocatedMemory = (m_iHeight+1)*m_iStep;
	}else{
		//keep the memory
		m_iWidth = w;	m_iHeight = h;	m_iChannel = channel;
		if(keep_old_step){
			if(m_iStep<new_step)	//cannot keep the old step
				return false;
		}else{
			m_iStep = new_step;
		}
	}

	if(pImage==NULL) return true;

	memset(m_pIImage, 0, sizeof(T)*m_iStep);
	T *line_sum = new T[m_iChannel];
	for(int y=1; y<=m_iHeight; y++){
		memset(m_pIImage+y*m_iStep, 0, sizeof(T)*m_iChannel);
		memset(line_sum, 0, sizeof(T)*m_iChannel);
		T* pII = m_pIImage + y*m_iStep + 1*m_iChannel;
		T* pIIEnd = pII + m_iWidth*m_iChannel;
		T* pIIPrevLine = m_pIImage + (y-1)*m_iStep + 1*m_iChannel;
		baseT* pI = pImage+(y-1)*step;
		while(pII!=pIIEnd){
			for(int c=0; c<m_iChannel; c++){
				line_sum[c] += *pI++;
				*pII++ = *(pIIPrevLine++) + line_sum[c];
			}
		}
	}
	delete []line_sum;

	return true;
}

template <class T, class baseT>
inline void TCIntegralImageMC<T, baseT>::Compute(int left, int top, int right, int bottom, T* rst) const
{
	T *pDes, *pSrc, *pSrcEnd;
	pDes = rst;	pSrc = m_pIImage + top*m_iStep + left*m_iChannel; pSrcEnd = pSrc+m_iChannel;
	while(pSrc!=pSrcEnd) *pDes++ = *pSrc++;
	pDes = rst;	pSrc = m_pIImage + top*m_iStep + right*m_iChannel; pSrcEnd = pSrc+m_iChannel;
	while(pSrc!=pSrcEnd) *pDes++ -= *pSrc++;
	pDes = rst;	pSrc = m_pIImage + bottom*m_iStep + left*m_iChannel; pSrcEnd = pSrc+m_iChannel;
	while(pSrc!=pSrcEnd) *pDes++ -= *pSrc++;
	pDes = rst;	pSrc = m_pIImage + bottom*m_iStep + right*m_iChannel; pSrcEnd = pSrc+m_iChannel;
	while(pSrc!=pSrcEnd) *pDes++ += *pSrc++;
}

//////////////////////////////////////////////////////////////////////////
//				Multi Channel Integral Image (Stride)					//
//////////////////////////////////////////////////////////////////////////


template <class T, class baseT>
class TCIntegralImageMCStride : public TCIntegralImageMC<T, baseT>
{
public:
	int m_iStride;

public:
	bool InitIntegralImage(int w, int h, int step, int channel, int stride, baseT* pImage, bool keep_old_step);				//compute integral image w.r.t a single image
	inline void Compute(int left, int top, int right, int bottom, T* rst) const;
	
};

template <class T, class baseT>
bool TCIntegralImageMCStride<T,baseT>::InitIntegralImage(int w, int h, int step, int channel, int stride, baseT* pImage, bool keep_old_step)
{
	m_iStride = stride;
	this->m_iWidth = w;
	this->m_iHeight = h;
	this->m_iChannel = channel;

	int nw = w/stride;
	int nh = h/stride;

	int new_step = (((nw+1)*channel*sizeof(T)+3)>>2<<2)/sizeof(T);
	if((nh+1)*new_step>this->m_iAllocatedMemory){
		//reallocate memory
		if(this->m_pIImage) delete []this->m_pIImage;
		this->m_iStep = new_step; 
		this->m_pIImage = new T[(nh+1)*this->m_iStep];
		if(this->m_pIImage==NULL) return false;
		this->m_iAllocatedMemory = (nh+1)*this->m_iStep;
	}else{
		//keep the memory
		if(keep_old_step){
			if(this->m_iStep<new_step)	//cannot keep the old step
				return false;
		}else{
			this->m_iStep = new_step;
		}
	}

	if(pImage==NULL) return true;

	memset(this->m_pIImage, 0, sizeof(T)*this->m_iStep);
	T *line_sum = new T[this->m_iChannel];
	for(int y=1; y<=nh; y++){
		memset(this->m_pIImage+y*this->m_iStep, 0, sizeof(T)*this->m_iChannel);
		memset(line_sum, 0, sizeof(T)*this->m_iChannel);
		for(int x=1; x<=nw; x++){
			for(int j=0; j<stride; j++){
				for(int i=0; i<stride; i++){
					baseT *p = pImage + ((y-1)*stride+j)*step + ((x-1)*stride+i)*channel;
					for(int c=0; c<channel; c++)
						line_sum[c] += p[c];
				}
			}
			for(int c=0; c<channel; c++)
				this->m_pIImage[y*this->m_iStep+x*channel+c] = this->m_pIImage[(y-1)*this->m_iStep+x*channel+c] + line_sum[c];
		}
	}
	delete []line_sum;

	return true;
}

template <class T, class baseT>
inline void TCIntegralImageMCStride<T, baseT>::Compute(int left, int top, int right, int bottom, T* rst) const
{
    if(left%m_iStride!=0 || right%m_iStride!=0 || top%m_iStride!=0 || bottom%m_iStride!=0){
        REPORT_ERROR_POSITION
    }

	left /= m_iStride;	right /= m_iStride;	top /= m_iStride;	bottom /= m_iStride;
	T *pDes, *pSrc, *pSrcEnd;
	pDes = rst;	pSrc = this->m_pIImage + top*this->m_iStep + left*this->m_iChannel; pSrcEnd = pSrc+this->m_iChannel;
	while(pSrc!=pSrcEnd) *pDes++ = *pSrc++;
	pDes = rst;	pSrc = this->m_pIImage + top*this->m_iStep + right*this->m_iChannel; pSrcEnd = pSrc+this->m_iChannel;
	while(pSrc!=pSrcEnd) *pDes++ -= *pSrc++;
	pDes = rst;	pSrc = this->m_pIImage + bottom*this->m_iStep + left*this->m_iChannel; pSrcEnd = pSrc+this->m_iChannel;
	while(pSrc!=pSrcEnd) *pDes++ -= *pSrc++;
	pDes = rst;	pSrc = this->m_pIImage + bottom*this->m_iStep + right*this->m_iChannel; pSrcEnd = pSrc+this->m_iChannel;
	while(pSrc!=pSrcEnd) *pDes++ += *pSrc++;
}

#endif

