
#ifndef __LP_INNER_HPP__
#define __LP_INNER_HPP__

#include "LPDetectRecog.hpp"

#include "thread/simple_thread_pool.h"


#define MAX_GP_NUMBER 64


#define ANCHOR_NUM 34
#define LP_SCORE_MAX 0.90f
#define LP_ROIP_SCORE_MAX 0.80f

#define MAX_RECOG_THREAD_NUM 1
#define DO_FCN_THREAD 1

typedef struct _LPDR_ImageInner_S {
    uchar *pubyData; //unsigned char data
    float *pfData; //float data
    int dwImgW; //image width
    int dwImgH; //image height
    int adwPRect[4]; //rectangle in parent
    int dwPID; //parent image ID
} LPDR_ImageInner_S;


struct LPRect {
  int dwX0;
  int dwY0;
  int dwX1;
  int dwY1;
};


struct LPPoint2D {
  int dwX;
  int dwY;
};


struct LPRectInfo {
	float fScore;
	float fCentX;
	float fCentY;
	float fWidth;
	float fHeight;
	LPRectInfo():fScore(0.0f), fCentX(0.0f), fCentY(0.0f),
						fWidth(0.0f), fHeight(0.0f) {}
	LPRectInfo(float score,
						float centy, float centx,
						float height, float width):
						fScore(score), fCentX(centx), fCentY(centy),
						fWidth(width), fHeight(height) {}
};


struct InputInfoRecog_S
{
  float *pfImage_0;
  float *pfImage_1;
  int dwH, dwW;
  LPRect rect;
  int dwSepY;
  char *pbyBuffer;
  int dwBufferLen;
};


struct LPDR_Info_S {
	LPDR_HANDLE hFCNN; //fcnn module
	
	LPDR_HANDLE hRPN; //rpn module
	
	LPDR_HANDLE hROIP; //region of interest pooling module

#if MAX_RECOG_THREAD_NUM>1
	LPDR_HANDLE ahPREGs[MAX_RECOG_THREAD_NUM]; //polygon regression module
	
	LPDR_HANDLE ahCHRECOGs[MAX_RECOG_THREAD_NUM]; //char recognition module
#else
  LPDR_HANDLE hPREG; //polygon regression module
	
	LPDR_HANDLE hCHRECOG; //char recognition module
#endif	
	size_t maxbuffer_size; 
	mx_float *pfBuffer; 
	
	uchar *pubyGrayImage;
	int dwGrayImgW;
	int dwGrayImgH;
	
	vector<LPRectInfo> *pvBBGroupOfROIP;
	vector<LPRectInfo> *pvBBGroupOfNMS;

	int dwDev_Type;
	int dwDev_ID;
	
	dg::ThreadPool *p_ppTPool;
	dg::ThreadPool *p_rfcnTPool;
	dg::ThreadPool *p_recogTPool;
};


extern int gdwAnchorBoxes[ANCHOR_NUM][2];

int copy_ndarray(NDArrayHandle hparam, NDArrayHandle hparamto, mx_float *pfBuffer, size_t buffersize);


int getSize(NDArrayHandle hout);


int group_bbs(vector<LPRectInfo> &lprects, vector<LPRectInfo> &group, float fiouThd);


int group_bbs_overlap(vector<LPRectInfo> &lprects, vector<LPRectInfo> &group, float fiouThd);


float calc_IOU(LPRectInfo &rect0, LPRectInfo &rect1);


int calc_overlap(LPRectInfo &rect0, LPRectInfo &rect1, float *pfOR0, float *pfOR1);


int getBestLPRect(mx_uint imgh, mx_uint imgw, mx_uint adims[2], mx_uint ashapes[2][4],
									mx_float *pfScore, int scoreSize,
									mx_float *pfRect, int rectSize,
									vector<LPRectInfo> &group);


void imgResizeAddBlack(uchar *patch, int s32W_src, int s32H_src,
													 uchar *tmpBuffer, uchar *result, 
													 int s32W_dst, int s32H_dst, int *pReal_w, int *pReal_h);


void imgResizeAddBlack_f(float *pfInputImg, int dwSrcW, int dwSrcH, float *pfDstImg, 
													 int dwDstW, int s32DstH, int *pdwRealW, int *pdwRealH);

void imgResizeAddBlack_fNorm(float *pfInputImg, int dwSrcW, int dwSrcH, float *pfDstImg, 
													 int dwDstW, int dwDstH, int *pdwRealW, int *pdwRealH);

void imgResize(uchar *patch, int s32W_src, int s32H_src, uchar *result, int s32W_dst, int s32H_dst);


void normalize_img_data(uchar *pubyImgData, int dwW, int dwH, int dwRatio);


int doNormContrastBB_f(float *pfImage, int dwH, int dwW, LPRect bb);


int calcNewMarginBB(int dwImgH, int dwImgW, LPRect *pstBB, int adwMRatioXY[2]);


int doRectify_f4(float *pfImage0, float *pfImage1, int dwW, int dwH, float fAngle_old, int adwPolygonXY[8], float *pfAngle_new);

int doRectify_f6(float *pfImage0, float *pfImage1, int dwW, int dwH, float fAngle_old, int adwPolygonXY[12], float *pfAngle_new);


int doRotate_f(float *pfImage, int dwW, int dwH, float fAngle);

int doRotate_8UC3(uchar *pubyImage, int dwW, int dwH, float fAngle);

int getMeanByHist(int *pdwHist, int dwLen);

int getBinThresholdIterByHist_uchar(uchar *pubyData, int dwLen);

int cvtRGB2HSV_U8(uchar ubyR, uchar ubyG, uchar ubyB, float *pfH, float *pfS, float *pfV);



#endif




