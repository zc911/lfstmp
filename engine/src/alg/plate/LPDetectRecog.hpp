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


#define LPDR_DBG 0
#define LPDR_TIME 0

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;

typedef void *LPDR_HANDLE;
typedef unsigned char uchar;

#define MAX_LP_NUM 16
#define MAX_LPCHAR_NUM 16
#define MAX_IMAGE_SET_MAXSIZE 512
#define MAX_SHAPE_SIZE 128
#define LPDR_CLASS_NUM 79


/*
chardict={u'blank':0, u'0':1, u'1':2, u'2':3, u'3':4, u'4':5, u'5':6, u'6':7, u'7':8, u'8':9, u'9':10, \
          u'A':11, u'B':12, u'C':13, u'D':14, u'E':15, u'F':16, u'G':17, u'H':18, u'J':19, \
          u'K':20, u'L':21, u'M':22, u'N':23, u'P':24, u'Q':25, u'R':26, u'S':27, u'T':28,\
          u'U':29, u'V':30, u'W':31, u'X':32, u'Y':33, u'Z':34, u'I':35, u'京':36, u'津':37,\
          u'沪':38, u'渝':39, u'冀':40, u'豫':41, u'云':42, u'辽':43, u'黑':44, u'湘':45, \
          u'皖':46, u'闽':47, u'鲁':48, u'新':49, u'苏':50, u'浙':51, u'赣':52, u'鄂':53, \
          u'桂':54, u'甘':55, u'晋':56, u'蒙':57, u'陕':58, u'吉':59, u'贵':60, u'粤':61, \
          u'青':62, u'藏':63, u'川':64, u'宁':65, u'琼':66, u'使':67, u'领':68, u'试':69, \
          u'学':70, u'临':71, u'时':72, u'警':73, u'港':74, u'O':75, u'挂':76, u'澳':77}


const char *paInv_chardict[79] = {"blank", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
          "A", "B", "C", "D", "E", "F", "G", "H", "J", \
          "K", "L", "M", "N", "P", "Q", "R", "S", "T",\
          "U", "V", "W", "X", "Y", "Z", "I", "京", "津",\
          "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘", \
          "皖", "闽", "鲁", "新", "苏", "浙", "赣", "鄂", \
          "桂", "甘", "晋", "蒙", "陕", "吉", "贵", "粤", \
          "青", "藏", "川", "宁", "琼", "使", "领", "试", \
          "学", "临", "时", "警", "港", "O", "挂", "澳", "#"};


const char *paInv_chardict_[79] = {"blank", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
          "A", "B", "C", "D", "E", "F", "G", "H", "J", \
          "K", "L", "M", "N", "P", "Q", "R", "S", "T",\
          "U", "V", "W", "X", "Y", "Z", "I", "_Jing1_", "_Jin1_",\
          "_Hu4_", "_Yu2_", "_Ji4_", "_Yu4_", "_Yun2_", "Liao2_", "_Hei1_", "_Xiang1_", \
          "_Wan3_", "_Min3_", "_Lu3_", "_Xin1_", "_Su1_", "_Zhe4_", "_Gan4_", "_E4_", \
          "_Gui4_", "_Gan1_", "_Jin4", "_Meng3_", "_Shan3_", "_Ji2_", "_Gui4_", "_Yue4_", \
          "_Qing1_", "_Zang4_", "_Chuan1_", "_Ning2_", "_Qiong2_", "_Shi3_", "_Ling3_", "_Shi4_", \
          "_Xue2_", "_Lin2_", "_Shi2_", "_Jing3_", "_Gang3_", "O", "_Gua2_", "_Ao4_", "#"};

*/




#define LP_COLOUR_WHITE     0
#define LP_COLOUR_SILVER    1
#define LP_COLOUR_YELLOW    2
#define LP_COLOUR_PINK      3
#define LP_COLOUR_RED       4
#define LP_COLOUR_GREEN        5
#define LP_COLOUR_BLUE        6
#define LP_COLOUR_BROWN        7
#define LP_COLOUR_BLACK        8

#define LP_TYPE_SINGLE 0
#define LP_TYPE_DOUBLE 1


typedef struct _LPDRModel {
    char *pbySym; //symbol model
    char *pbyParam; //model parameter
    int dwParamSize; //model parameter size
    int adwShape[MAX_SHAPE_SIZE]; //batch number, channel number, height, width, ...
} LPDRModel_S;


typedef struct _LPDRConfig {
    LPDRModel_S stFCNN; //Full CNN, note:dwImageNum of LPDR_ImageSet_S can't be larger than dwBatchSize of stFCNN.
    LPDRModel_S stRPN; //Region Proposal
    LPDRModel_S stROIP; //Region of Interest Pooling
    LPDRModel_S stPREG; //Polygon Regression
    LPDRModel_S stCHRECOG; //Recognition
    int dwDevType; //device type: {'cpu': 1, 'gpu': 2, 'cpu_pinned': 3}
    int dwDevID; //device ID: 0, 1, ...
    string fcnnSymbolFile;
    string fcnnParamFile;
    string rpnSymbolFile;
    string rpnParamFile;
    string roipSymbolFile;
    string roipParamFile;
    string pregSymbolFile;
    string pregParamFile;
    string chrecogSymbolFile;
    string chrecogParamFile;
    int batchsize = 1;
    bool is_model_encrypt = true;
    int imageSW;
    int imageSH;
    int numsPlates;
    int plateSW;
    int plateSH;
    int numsProposal;
} LPDRConfig_S;


typedef struct _LPDRInfo {
    int adwLPRect[4]; //top, left, right, bottom
    int adwLPPolygon[8]; //pnt0(x,y), pnt1(x,y), pnt2(x,y), pnt3(x,y)
    int adwLPNumber[MAX_LPCHAR_NUM]; //number name of char
    float afScores[MAX_LPCHAR_NUM]; //score for eatch char
    int dwLPLen; //chars number
    float fAllScore; //LP score
    int dwColor; //LP color
    int dwType; //LP type, single line, or double line
} LPDRInfo_S;


typedef struct _LPDR_Output {
    LPDRInfo_S astLPs[MAX_LP_NUM]; //LP set
    int dwLPNum; //LP Number
} LPDR_Output_S;


typedef struct _LPDR_OutputSet {
    LPDR_Output_S astLPSet[MAX_IMAGE_SET_MAXSIZE]; //LPs for each image
    int dwImageNum; //image number
} LPDR_OutputSet_S;


typedef struct _LPDR_Image {
    uchar *pubyData; // RGB or YUV data of Image
    int dwImgW; //image width
    int dwImgH; //image height
} LPDR_Image_S;


typedef struct _LPDR_ImageSet {
    LPDR_Image_S astSet[MAX_IMAGE_SET_MAXSIZE]; //image set
    int dwImageNum; //image number
} LPDR_ImageSet_S;


int LPDR_Create(LPDR_HANDLE *pHandle, LPDRConfig_S *pstConfig);

int LPDR_Process(LPDR_HANDLE handle, LPDR_ImageSet_S *pstImgSet, LPDR_OutputSet_S *pstOutputSet);

int LPDR_Release(LPDR_HANDLE handle);


#endif




