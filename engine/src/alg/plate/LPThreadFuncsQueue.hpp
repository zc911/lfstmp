
#ifndef __LP_THREAD_FUNCS_QUEUE_HPP__
#define __LP_THREAD_FUNCS_QUEUE_HPP__

#include "LPDetectRecog.hpp"
#include "LPInner.hpp"
#include "LPFCNN.hpp"


int lpPreProcessThreadsQueue(dg::ThreadPool *p_ppTPool, LPDR_ImageSet_S *pstImgSet, LPDR_ImageInner_S *pstFCNNImgSet);
int lpReadyFCNDataThreadsQueue(dg::ThreadPool *p_rfcnTPool, LPDR_ImageInner_S *pstImgSet, int dwImgNum, ModuleFCNN_S *pstFCNN);
int doRecognitions_ThreadsQueue(dg::ThreadPool *p_recogTPool, LPDR_HANDLE handle, LPDR_ImageInner_S *pstImgSet, int dwImgNum, LPDR_OutputSet_S *pstOutputSet);


#endif



