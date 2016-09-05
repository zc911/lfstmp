#ifndef __LP_DETECTRECOG_HPP__
#define __LP_DETECTRECOG_HPP__


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include "c_api.h"
#include "math.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define LPDR_DBG 0
#define LPDR_TIME 0
using namespace std;

typedef void* LPDR_HANDLE;
typedef unsigned char uchar;

#define MAX_LP_NUM 16
#define MAX_LPCHAR_NUM 16
#define MAX_IMAGE_SET_MAXSIZE 512
#define MAX_SHAPE_SIZE 128

#define LP_COLOUR_UNKNOWN   0
#define LP_COLOUR_BLUE      1
#define LP_COLOUR_YELLOW    2
#define LP_COLOUR_WHITE     3
#define LP_COLOUR_BLACK     4
#define LP_COLOUR_GREEN     5

#define LP_TYPE_SINGLE 0
#define LP_TYPE_DOUBLE 1


typedef struct _LPDRModel {
    char *pbySym; //symbol model
    char *pbyParam; //model parameter
    int dwParamSize; //model parameter size
    int adwShape[MAX_SHAPE_SIZE]; //batch number, channel number, height, width, ...
}LPDRModel_S;


typedef struct _LPDRConfig {
  LPDRModel_S stFCNN; //Full CNN, note:dwImageNum of LPDR_ImageSet_S can't be larger than dwBatchSize of stFCNN.
  LPDRModel_S stRPN; //Region Proposal
  LPDRModel_S stROIP; //Region of Interest Pooling
  LPDRModel_S stPREG; //Polygon Regression
  LPDRModel_S stCHRECOG; //Recognition
  LPDRModel_S stCOLOR; //Color Recognition
  int dwDevType; //device type: {'cpu': 1, 'gpu': 2, 'cpu_pinned': 3}
  int dwDevID; //device ID: 0, 1, ...
}LPDRConfig_S;


typedef struct _LPDRInfo {
  int adwLPRect[4]; //top, left, right, bottom
  int adwLPPolygon[8]; //pnt0(x,y), pnt1(x,y), pnt2(x,y), pnt3(x,y)
  int adwLPNumber[MAX_LPCHAR_NUM]; //number name of char
  float afScores[MAX_LPCHAR_NUM]; //score for eatch char
  int dwLPLen; //chars number
  float fAllScore; //LP score
  int dwColor; //LP color
  int dwType; //LP type, single line, or double line
}LPDRInfo_S;


typedef struct _LPDR_Output {
  LPDRInfo_S astLPs[MAX_LP_NUM]; //LP set
  int dwLPNum; //LP Number
}LPDR_Output_S;


typedef struct _LPDR_OutputSet {
  LPDR_Output_S astLPSet[MAX_IMAGE_SET_MAXSIZE]; //LPs for each image
  int dwImageNum; //image number
}LPDR_OutputSet_S;


typedef struct _LPDR_Image {
    uchar *pubyData; // RGB or YUV data of Image
    int dwImgW; //image width
    int dwImgH; //image height
}LPDR_Image_S;


typedef struct _LPDR_ImageSet {
    LPDR_Image_S astSet[MAX_IMAGE_SET_MAXSIZE]; //image set
    int dwImageNum; //image number
}LPDR_ImageSet_S;


int LPDR_Create(LPDR_HANDLE *pHandle, LPDRConfig_S *pstConfig);

int LPDR_Process(LPDR_HANDLE handle, LPDR_ImageSet_S *pstImgSet, LPDR_OutputSet_S *pstOutputSet);

int LPDR_Release(LPDR_HANDLE handle);


#endif









