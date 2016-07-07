#ifndef __LP_PREG_HPP__
#define __LP_PREG_HPP__


#include "LPDetectRecog.hpp"
#include  "LPInner.hpp"


struct ModulePREG_S {
    ExecutorHandle hExecute;
    NDArrayHandle *args_arr;
    int args_num = 0;
    SymbolHandle hSymbol;

    int adwInShape[4]; //input image shape
    int adwOutShape[2]; //polygon shape

    float *pfStdData;
    float *pfOutPolygon;
};


int LPPREG_Create(LPDRModel_S stPREG, int dwDevType, int dwDevID, LPDR_HANDLE *phPREG);

int LPPREG_Process(LPDR_HANDLE hPREG, LPDR_ImageInner_S *pstImage, int adwPolygonOut[12]);

int LPPREG_Release(LPDR_HANDLE hPREG);


#endif



