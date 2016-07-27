
#ifndef __LP_THREAD_FUNCS_HPP__
#define __LP_THREAD_FUNCS_HPP__

#include "LPDetectRecog.hpp"
#include "LPInner.hpp"
#include "pthread.h"
#include "LPFCNN.hpp"


int doRecognitions_Threads(LPDR_HANDLE handle, LPDR_ImageInner_S *pstImgSet, int dwImgNum, LPDR_OutputSet_S *pstOutputSet);
int lpPreProcessThreads(LPDR_ImageSet_S *pstImgSet, LPDR_ImageInner_S *pstFCNNImgSet);
int lpReadyFCNDataThreads(LPDR_ImageInner_S *pstImgSet, int dwImgNum, ModuleFCNN_S *pstFCNN);

#endif



