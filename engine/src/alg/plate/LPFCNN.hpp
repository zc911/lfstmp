#ifndef LP_FCNN_HPP
#define LP_FCNN_HPP


#include "LPDetectRecog.hpp"
#include "LPInner.hpp"


struct ModuleFCNN_S {
    ExecutorHandle hExecute;
    NDArrayHandle *args_arr;
    int args_num = 0;
    SymbolHandle hSymbol;

    int adwInShape[4]; //input image shape
    int adwOutShape[4]; //class map shape

    uchar *pubyInputData;
    float *pfInputData;

    float *pfOutputData;

    uchar *pubyBuffer;
    int dwBuffSize;

    int *pdwRects;
    int dwRectNum;

    int dwCheckW;
    int dwCheckH;

    vector<LPRectInfo> *plpgroup;

    int *pdwRealWs;
    int *pdwRealHs;
};


int LPFCNN_Create(LPDRModel_S stFCNN, int dwDevType, int dwDevID, LPDR_HANDLE *phFCNN);

int LPFCNN_Process(LPDR_HANDLE hFCNN, LPDR_ImageInner_S *pstImgSet, int dwImgNum);

int LPFCNN_Release(LPDR_HANDLE hFCNN);


#endif



