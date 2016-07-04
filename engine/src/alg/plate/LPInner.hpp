
#ifndef __LP_INNER_HPP__
#define __LP_INNER_HPP__

#include "LPDetectRecog.hpp"


#define MAX_GP_NUMBER 64


#define ANCHOR_NUM 34
#define LP_SCORE_MAX 0.90f

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


extern int gdwAnchorBoxes[ANCHOR_NUM][2];

int copy_ndarray(NDArrayHandle hparam, NDArrayHandle hparamto, mx_float *pfBuffer, size_t buffersize);


int getSize(NDArrayHandle hout);


int group_bbs(vector<LPRectInfo> &lprects, vector<LPRectInfo> &group, float fiouThd);


float calc_IOU(LPRectInfo &rect0, LPRectInfo &rect1);


int getBestLPRect(mx_uint imgh, mx_uint imgw, mx_uint adims[2], mx_uint ashapes[2][4],
									mx_float *pfScore, int scoreSize,
									mx_float *pfRect, int rectSize,
									vector<LPRectInfo> &group);


void imgResizeAddBlack(uchar *patch, int s32W_src, int s32H_src,
													 uchar *tmpBuffer, uchar *result, 
													 int s32W_dst, int s32H_dst, int *pReal_w, int *pReal_h);


void imgResizeAddBlack_f(float *pfInputImg, int dwSrcW, int dwSrcH, float *pfDstImg, 
													 int dwDstW, int s32DstH, int *pdwRealW, int *pdwRealH);


void imgResize(uchar *patch, int s32W_src, int s32H_src, uchar *result, int s32W_dst, int s32H_dst);


void normalize_img_data(uchar *pubyImgData, int dwW, int dwH, int dwRatio);


int doNormContrastBB_f(float *pfImage, int dwH, int dwW, LPRect bb);


int calcNewMarginBB(int dwImgH, int dwImgW, LPRect *pstBB, int adwMRatioXY[2]);


int doRectify_f(float *pfImage0, float *pfImage1, int dwW, int dwH, float fAngle_old, int adwPolygonXY[8], float *pfAngle_new);


int doRotate_f(float *pfImage, int dwW, int dwH, float fAngle);

int doRotate_8UC3(uchar *pubyImage, int dwW, int dwH, float fAngle);

int getMeanByHist(int *pdwHist, int dwLen);

int getBinThresholdIterByHist_uchar(uchar *pubyData, int dwLen);


#endif




