
#ifndef __LP_CH_RECOG_HPP__
#define __LP_CH_RECOG_HPP__

#include "LPDetectRecog.hpp"
#include  "LPInner.hpp"

#define MAX_LP_CHRECOG_NUM 8


struct ModuleCHRECOG_S {
	ExecutorHandle hExecute;
	NDArrayHandle *args_arr;
	int args_num = 0; 
	SymbolHandle hSymbol;

	int adwInShape[4]; //input image shape
	int adwOutShape[2]; //class shape
	
	float *pfInData;

	float *pfOutScore;
	int *pdwClassIdx;
	float *pfClassScore;
	
	char *pbyBuffer;
	int dwBufferSZ;
};


int LPCHRECOG_Create(LPDRModel_S stCHRECOG, int dwDevType, int dwDevID, LPDR_HANDLE *phCHRECOG);

int LPCHRECOG_Process(LPDR_HANDLE hCHRECOG, LPDR_ImageInner_S *pstImage, LPRect rect, float fStrechRatio, float fShrinkRatio, int dwStep);

int LPCHRECOG_Release(LPDR_HANDLE hCHRECOG);


#endif

