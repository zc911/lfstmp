
#ifndef __LP_COLOR_HPP__
#define __LP_COLOR_HPP__

#include "LPDetectRecog.hpp"
#include  "LPInner.hpp"


struct ModuleCOLOR_S {
	ExecutorHandle hExecute;
	NDArrayHandle *args_arr;
	int args_num = 0; 
	SymbolHandle hSymbol;

	int adwInShape[4]; //input image shape
	int adwOutShape[1]; //color shape
	
	float *pfInData;

	float *pfOutScore;
	int *pdwClassIdx;
	float *pfClassScore;
	
	char *pbyBuffer;
	int dwBufferSZ;
};


int LPCOLOR_Create(LPDRModel_S stCOLOR, int dwDevType, int dwDevID, LPDR_HANDLE *phCOLOR);

int LPCOLOR_Process(LPDR_HANDLE hCOLOR, float *pfData, int dwDH, int dwDW, int *pdwColor);

int LPCOLOR_Release(LPDR_HANDLE hCOLOR);


#endif

