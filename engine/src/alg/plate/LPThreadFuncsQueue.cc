
#include "LPThreadFuncsQueue.hpp"



int doRecogOne(LPDR_HANDLE hPolyReg, LPDR_HANDLE hChRecog, InputInfoRecog_S *pstIIR, LPDRInfo_S *pstOut);


#if DO_FCN_THREAD

struct LP_RFCND_THREAD_S {
  LPDR_ImageInner_S *pstImage;
  float *pfStdInputData;
  int dwStdH;
  int dwStdW;
  int *pdwRealW;
  int *pdwRealH;
  condition_variable *p_cv;
  mutex *p_countmt;
  int *p_dwFinishCount;
  int dwNeedFinishNum;
};



void *lpReadyFCNDataThreadQueueOne(void *pParam);

int lpReadyFCNDataThreadsQueue(dg::ThreadPool *p_TPool, LPDR_ImageInner_S *pstImgSet, int dwImgNum, ModuleFCNN_S *pstFCNN)
{
  int dwTI;
  LP_RFCND_THREAD_S astParams[256];
  mutex countmt;
  condition_variable cv;
  int dwStdH, dwStdW;
  int dwFinishCount = 0;
  int dwNeedFinishNum = 0;//dwImgNum;

  assert(dwImgNum <= 256);

  for (dwTI = 0; dwTI < dwImgNum; dwTI++)
  {
    if (pstImgSet[dwTI].pfData)
    {
      dwNeedFinishNum++;
    }
  }
  
  memset(astParams, 0, sizeof(LP_RFCND_THREAD_S)*256);
  dwStdH = pstFCNN->adwInShape[2];
  dwStdW = pstFCNN->adwInShape[3];
  for (dwTI = 0; dwTI < dwImgNum; dwTI++)
  {
    LPDR_ImageInner_S *pstImage = &pstImgSet[dwTI];
//    printf("[%d]pstImage->pfData:%x\n", dwTI, pstImage->pfData);
    if (!pstImage->pfData)
    {
      continue;
    }
    astParams[dwTI].pfStdInputData = pstFCNN->pfInputData + dwTI * dwStdW * dwStdH;
    astParams[dwTI].pdwRealW = &pstFCNN->pdwRealWs[dwTI];
    astParams[dwTI].pdwRealH = &pstFCNN->pdwRealHs[dwTI];
    astParams[dwTI].dwStdW = dwStdW;
    astParams[dwTI].dwStdH = dwStdH;
    astParams[dwTI].pstImage = pstImage;
    astParams[dwTI].p_cv = &cv;
    astParams[dwTI].p_countmt = &countmt;
    astParams[dwTI].p_dwFinishCount = &dwFinishCount;
    astParams[dwTI].dwNeedFinishNum = dwNeedFinishNum;

    p_TPool->enqueue(lpReadyFCNDataThreadQueueOne, (void*)&astParams[dwTI]);
  }

  unique_lock<mutex> waitlc(countmt);
//  cv.wait_for(waitlc, chrono::seconds(1), [&dwFinishCount, &dwNeedFinishNum]() { return dwFinishCount == dwNeedFinishNum;});
  cv.wait(waitlc, [&dwFinishCount, &dwNeedFinishNum]() { return dwFinishCount == dwNeedFinishNum;});

  return 0;
}


void *lpReadyFCNDataThreadQueueOne(void *pParam)
{
  int dwRealW, dwRealH;
  
  LP_RFCND_THREAD_S *pstParam = (LP_RFCND_THREAD_S*)pParam;
  LPDR_ImageInner_S *pstImage = pstParam->pstImage;
  float *pfStdInputData = pstParam->pfStdInputData;
  int dwStdH = pstParam->dwStdH;
  int dwStdW = pstParam->dwStdW;
  int *pdwRealH = pstParam->pdwRealH;
  int *pdwRealW = pstParam->pdwRealW;
  
  condition_variable *p_cv = pstParam->p_cv;
  mutex *p_countmt = pstParam->p_countmt;
  int *p_dwFinishCount = pstParam->p_dwFinishCount;
  int dwNeedFinishNum = pstParam->dwNeedFinishNum;

  float *pfDataOri = pstImage->pfData;
  int dwImgWOri = pstImage->dwImgW;
  int dwImgHOri = pstImage->dwImgH;
  
  memset(pfStdInputData, 0, sizeof(float) * dwStdW * dwStdH);
  imgResizeAddBlack_f(pfDataOri, dwImgWOri, dwImgHOri, pfStdInputData, dwStdW, dwStdH, &dwRealW, &dwRealH);
  
  *pdwRealH = dwRealH;
  *pdwRealW = dwRealW;
  
  int dwFinishCount = 0;
  unique_lock<mutex> countlc(*p_countmt);
  if ((*p_dwFinishCount) < dwNeedFinishNum) {
    (*p_dwFinishCount)++;
  }
  dwFinishCount = (*p_dwFinishCount);

  if (dwFinishCount == dwNeedFinishNum) {
    p_cv->notify_all();
  }
 
  countlc.unlock();
  
 
  return 0;
}


struct LP_PP_THREAD_S {
  LPDR_Image_S *pstOneIn;
  LPDR_ImageInner_S *pstOne;
  condition_variable *p_cv;
  mutex *p_countmt;
  int *p_dwFinishCount;
  int dwNeedFinishNum;
};

void *lpPreProcessThreadQueueOne(void *pParam);

int lpPreProcessThreadsQueue(dg::ThreadPool *p_TPool, LPDR_ImageSet_S *pstImgSet, LPDR_ImageInner_S *pstFCNNImgSet)
{
  int dwI;
  int dwImgNum = pstImgSet->dwImageNum;
  LP_PP_THREAD_S astPPs[256];
  mutex countmt;
  condition_variable cv;
  int dwFinishCount = 0;
  int dwNeedFinishNum = dwImgNum;

  assert(dwImgNum <= 256);

  memset(astPPs, 0, 256*sizeof(LP_PP_THREAD_S));
  
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    LPDR_Image_S *pstImgIn = &pstImgSet->astSet[dwI];
    LPDR_ImageInner_S *pstOne = &pstFCNNImgSet[dwI];
    astPPs[dwI].pstOneIn = pstImgIn;
    astPPs[dwI].pstOne = pstOne;
    astPPs[dwI].p_cv = &cv;
    astPPs[dwI].p_countmt = &countmt;
    astPPs[dwI].p_dwFinishCount = &dwFinishCount;
    astPPs[dwI].dwNeedFinishNum = dwNeedFinishNum;
    
    p_TPool->enqueue(lpPreProcessThreadQueueOne, (void*)&astPPs[dwI]);
//    printf("%d:%d\n", dwI, p_TPool->size());
  }
  

  unique_lock<mutex> waitlc(countmt);
//  cv.wait_for(waitlc, chrono::seconds(1), [&dwFinishCount, &dwNeedFinishNum]() { return dwFinishCount == dwNeedFinishNum;});
  cv.wait(waitlc, [&dwFinishCount, &dwNeedFinishNum]() {return dwFinishCount == dwNeedFinishNum;});
//  printf("ok now!\n");

  return 0;
}


void *lpPreProcessThreadQueueOne(void *pParam)
{
  LP_PP_THREAD_S *pstPP = (LP_PP_THREAD_S*)pParam;
  LPDR_Image_S *pstOneIn = pstPP->pstOneIn;
  LPDR_ImageInner_S *pstOne = pstPP->pstOne;

//  printf("lpPreProcessThreadQueueOne_0\n");
  
  condition_variable *p_cv = pstPP->p_cv;
  mutex *p_countmt = pstPP->p_countmt;
  int *p_dwFinishCount = pstPP->p_dwFinishCount;
  int dwNeedFinishNum = pstPP->dwNeedFinishNum;
#if 1  
  int dwImgW = pstOneIn->dwImgW;
  int dwImgH = pstOneIn->dwImgH;
  pstOne->dwImgW = dwImgW;
  pstOne->dwImgH = dwImgH;
  pstOne->pfData = new float[pstOne->dwImgW * pstOne->dwImgH];
  int dwSize = pstOne->dwImgW * pstOne->dwImgH;
  
//  printf("lpPreProcessThreadQueueOne_1\n");

  cv::Mat inputColorOne(dwImgH, dwImgW, CV_8UC3, pstOneIn->pubyData);
  cv::Mat inputGrayOne(dwImgH, dwImgW, CV_8UC1);
  cv::cvtColor(inputColorOne, inputGrayOne, CV_BGR2GRAY);

//  printf("lpPreProcessThreadQueueOne_2\n");

  uchar *pubyOne = (uchar*)inputGrayOne.data;
  cv::Mat oneData(dwImgH, dwImgW, CV_32FC1, pstOne->pfData);
  inputGrayOne.convertTo(oneData, CV_32FC1, 1.0f/255.f, 0);  
#endif
  int dwFinishCount = 0;
//  printf("dwFinishCount_0:%d/%d\n", dwFinishCount, dwNeedFinishNum); 
#if 1
  unique_lock<mutex> countlc(*p_countmt);
  if ((*p_dwFinishCount) < dwNeedFinishNum) {
    (*p_dwFinishCount)++;
  }
  dwFinishCount = (*p_dwFinishCount);
//  printf("dwFinishCount_1:%d\n", dwFinishCount); 
//  printf("dwFinishCount_1:%d/%d\n", dwFinishCount, dwNeedFinishNum); 
  if (dwFinishCount == dwNeedFinishNum)
  {
//    printf("dwFinishCount_2:%d\n", dwFinishCount); 
    p_cv->notify_all();
  }

  countlc.unlock();
#endif  


  return 0;
}


#endif


#if MAX_RECOG_THREAD_NUM>1
struct REG_RECOG_MISSION_S {
  float *pfMomImage;
  int dwImgW;
  int dwImgH;
  LPDR_Output_S *pstLPDRSetOne; //need to lock
  LPRectInfo stLPRect;
};


struct REG_RECOG_GLOBAL_S {
  vector<REG_RECOG_MISSION_S> *pvecMission; //lock
  int dwNowMissionID; //lock
  int dwMissionNum;
	int dwDev_Type;
	int dwDev_ID;
	
	int *p_dwFinishCount;
  int dwNeedFinishNum;

  condition_variable *p_cv;
  mutex *p_countmt;
  mutex *p_missionmt;
};

struct REG_RECOG_S {
  LPDR_HANDLE hPREG;
  LPDR_HANDLE hCHRECOG;
  
  int dwThreadID;

  REG_RECOG_GLOBAL_S *pstGlobal;
};


void *doRecogOne_ThreadQueue(void *pParams);
int doRecognitions_ThreadsQueue(dg::ThreadPool *p_TPool, LPDR_HANDLE handle, LPDR_ImageInner_S *pstImgSet, int dwImgNum, LPDR_OutputSet_S *pstOutputSet)
{
#if LPDR_TIME&1
  float costtime, diff;
  struct timeval start, end;
  
  gettimeofday(&start, NULL);
#endif
  int dwI, dwJ;
  LPDR_Info_S *pstLPDR = (LPDR_Info_S*)handle;
  

  vector<LPRectInfo> *plproipnms = pstLPDR->pvBBGroupOfNMS;

  memset(pstOutputSet, 0, sizeof(LPDR_OutputSet_S));
  
  vector<REG_RECOG_MISSION_S> vecMissions;
  vecMissions.clear();

  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    int dwImgW = pstImgSet[dwI].dwImgW;
    int dwImgH = pstImgSet[dwI].dwImgH;
    float *pfImage = pstImgSet[dwI].pfData;
    LPDR_Output_S *pstLPDRSetOne = &pstOutputSet->astLPSet[dwI];
#if LPDR_DBG&0
    {
      cv::Mat gimg(dwImgH, dwImgW, CV_32FC1, pfImage);
      cv::imshow("gimg", gimg);
      cv::waitKey(0);
    }
#endif
    vector<LPRectInfo> &lproipnms_one = plproipnms[dwI];
    int dwSize = lproipnms_one.size();

    for (dwJ = 0; dwJ < dwSize; dwJ++)
    {
#if LPDR_DBG
      cout << "--------------------\n";
#endif
      REG_RECOG_MISSION_S stOne;
      stOne.pfMomImage = pfImage;
      stOne.dwImgW = dwImgW;
      stOne.dwImgH = dwImgH;
      stOne.stLPRect = lproipnms_one[dwJ];
      stOne.pstLPDRSetOne = pstLPDRSetOne;
      vecMissions.push_back(stOne);
      
//      printf("imgh:%d, imgw:%d\n", dwImgH, dwImgW);
    }
  }
  
  mutex countmt, missionmt;
  condition_variable cv;

  int dwMissionNum = vecMissions.size();
  
  REG_RECOG_S astParams[MAX_RECOG_THREAD_NUM];
  
  REG_RECOG_GLOBAL_S stGlobal;
  
  int dwFinishCount = 0;
  int dwNeedThreadNum = min(dwMissionNum, MAX_RECOG_THREAD_NUM);
  int dwNeedFinishNum = dwNeedThreadNum;
  
  memset(astParams, 0, sizeof(REG_RECOG_S)*MAX_RECOG_THREAD_NUM);

  stGlobal.pvecMission = &vecMissions;
  stGlobal.p_countmt = &countmt;
  stGlobal.p_missionmt = &missionmt;
  stGlobal.p_cv = &cv;
  stGlobal.dwNowMissionID = 0;
  stGlobal.dwMissionNum = dwMissionNum;
  stGlobal.dwDev_Type = pstLPDR->dwDev_Type;
	stGlobal.dwDev_ID = pstLPDR->dwDev_ID;
	stGlobal.p_dwFinishCount = &dwFinishCount;
  stGlobal.dwNeedFinishNum = dwNeedFinishNum;
  
//  printf("stGlobal.p_dwFinishCount:%d\n", *stGlobal.p_dwFinishCount);
  for (int dwTI = 0; dwTI < dwNeedThreadNum; dwTI++)
  {
//    printf("start new thread %d\n", dwTI);
    REG_RECOG_S *pstParam = &astParams[dwTI];
    pstParam->pstGlobal = &stGlobal;
    pstParam->hPREG = pstLPDR->ahPREGs[dwTI];
    pstParam->hCHRECOG = pstLPDR->ahCHRECOGs[dwTI];
    pstParam->dwThreadID = dwTI;
    
    p_TPool->enqueue(doRecogOne_ThreadQueue, (void*)pstParam);
  }
  
  unique_lock<mutex> waitlc(countmt);
//  cv.wait_for(waitlc, [&stGlobal]() { return stGlobal.dwNowMissionID == stGlobal.dwMissionNum;});
  cv.wait(waitlc, [&dwFinishCount, &dwNeedFinishNum]() { return dwFinishCount == dwNeedFinishNum;});

  pstOutputSet->dwImageNum = dwImgNum;

#if LPDR_TIME&1
  gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec) / 1000.f;
	printf("doRecognitions threads cost:%.2fms\n", diff);
#endif
  return 0;
}



void *doRecogOne_ThreadQueue(void *pParam)
{
  REG_RECOG_S *pstParam = (REG_RECOG_S*)pParam;
  LPDR_HANDLE hPREG = pstParam->hPREG;
  LPDR_HANDLE hCHRECOG = pstParam->hCHRECOG;
  int dwThreadID = pstParam->dwThreadID;
  REG_RECOG_GLOBAL_S *pstGlobal = pstParam->pstGlobal;
	int dwDev_Type = pstGlobal->dwDev_Type;
	int dwDev_ID = pstGlobal->dwDev_ID;
  int dwRI, dwLPI;
  int dwX0_0, dwX1_0, dwY0_0, dwY1_0, dwW_0, dwH_0;
  int dwX0_1, dwX1_1, dwY0_1, dwY1_1;
  int adwMarginHW[2];
  float *pfBlkBuffer_0 = 0, *pfBlkBuffer_1 = 0;
  char *pbyBuffer = 0;
  int dwMissionNum = pstGlobal->dwMissionNum;
  vector<REG_RECOG_MISSION_S> *pvecMission = pstGlobal->pvecMission;
  condition_variable *p_cv = pstGlobal->p_cv;
  mutex *p_countmt = pstGlobal->p_countmt;
  mutex *p_missionmt = pstGlobal->p_missionmt;
  
//  printf("aaaaaaaaaaaaaaaa!!!%d\n", dwThreadID);
  
  
  int dwBlkMaxLen = 1000 * 1000, dwBlkH, dwBlkW, dwBufferLen = 1000 * 1000 * 4;
  int dwImgW, dwImgH;
  float *pfImage = 0;
  
  float afMarginRatioHW[2] = {0.4f, 0.4f};
  float afCropSize[2] = {1.6f, 2.0f};
  float afNewSize[2] = {1.2f, 1.2f};
  
	if (dwDev_Type==2)
	{
	  cudaSetDevice(dwDev_ID);
	}

  LPDR_Output_S *pstLPDRSetOne = 0;
  int dwNowMissionID = -1;
  
  LPRectInfo lprect;

  pfBlkBuffer_0 = new float[dwBlkMaxLen];
  pfBlkBuffer_1 = new float[dwBlkMaxLen];
  pbyBuffer = new char[dwBufferLen];
  
//  printf("mission_0=>imgh:%d, imgw:%d\n", (*pvecMission)[0].dwImgH, (*pvecMission)[0].dwImgW);
  
  while (1) {
    //read data
    unique_lock<mutex> countlc1(*p_missionmt);
//    p_missionmt->lock();
    
//    printf("fucking new dwNowMissionID %x %d/%d[%d]\n", &pstGlobal->dwNowMissionID, pstGlobal->dwNowMissionID, dwMissionNum, dwThreadID);
    if (pstGlobal->dwNowMissionID >= dwMissionNum)
    {
//      printf("fucking thread[exit while]:%d/%d, %d\n", pstGlobal->dwNowMissionID, dwMissionNum, dwThreadID);
      countlc1.unlock();
//      p_missionmt->unlock();
      break;
    }

    REG_RECOG_MISSION_S &stMission = (*pvecMission)[pstGlobal->dwNowMissionID];
    pstGlobal->dwNowMissionID++;
    dwNowMissionID = pstGlobal->dwNowMissionID;
    
//    printf("start new mission %x %d/%d[%d]\n", &pstGlobal->dwNowMissionID, pstGlobal->dwNowMissionID, dwMissionNum, dwThreadID);
    
    pfImage = stMission.pfMomImage;
    dwImgW = stMission.dwImgW;
    dwImgH = stMission.dwImgH;
    lprect = stMission.stLPRect;
    pstLPDRSetOne = stMission.pstLPDRSetOne;

    countlc1.unlock();
//    p_missionmt->unlock();
    
    //process data
    LPRectInfo lprect_crop(lprect.fScore, lprect.fCentY, lprect.fCentX, lprect.fHeight*afCropSize[1], lprect.fWidth*afCropSize[0]);

    LPDRInfo_S stOut;
    InputInfoRecog_S stIIR;
    
    memset(&stOut, 0, sizeof(LPDRInfo_S));
    memset(&stIIR, 0, sizeof(InputInfoRecog_S));
    
    stIIR.pbyBuffer = pbyBuffer;
    stIIR.dwBufferLen = dwBufferLen;
    
    dwX0_0 = max(lprect_crop.fCentX - lprect_crop.fWidth/2, 0.0f);
    dwY0_0 = max(lprect_crop.fCentY - lprect_crop.fHeight/2, 0.0f);
    dwX1_0 = min(lprect_crop.fCentX + lprect_crop.fWidth/2, dwImgW-1.f);
    dwY1_0 = min(lprect_crop.fCentY + lprect_crop.fHeight/2, dwImgH-1.f);
    dwW_0 = dwX1_0 - dwX0_0 + 1;
    dwH_0 = dwY1_0 - dwY0_0 + 1;

    //add margin
    adwMarginHW[0] = (int)((dwH_0 * afMarginRatioHW[0])/2);
    adwMarginHW[1] = (int)((dwW_0 * afMarginRatioHW[1])/2);

    dwBlkH = adwMarginHW[0] * 2 + dwH_0;
    dwBlkW = adwMarginHW[1] * 2 + dwW_0;

//    assert(dwBlkH * dwBlkW < dwBlkMaxLen);
    if (dwBlkH * dwBlkW > dwBlkMaxLen) continue;

    memset(pfBlkBuffer_0, 0, sizeof(float) * dwBlkH * dwBlkW);
    for (dwRI = 0; dwRI < dwH_0; dwRI++)
    {
      memcpy(pfBlkBuffer_0 + (dwRI + adwMarginHW[0]) * dwBlkW + adwMarginHW[1], pfImage + (dwRI + dwY0_0) * dwImgW + dwX0_0, sizeof(float) * dwW_0);
    }

    LPRectInfo lprect_new(lprect.fScore, lprect.fCentY, lprect.fCentX, lprect.fHeight*afNewSize[1], lprect.fWidth*afNewSize[0]);
    dwX0_1 = max(lprect_new.fCentX - lprect_new.fWidth/2, 0.f);
    dwY0_1 = max(lprect_new.fCentY - lprect_new.fHeight/2, 0.f);
    dwX1_1 = min(lprect_new.fCentX + lprect_new.fWidth/2, dwImgW-1.f);
    dwY1_1 = min(lprect_new.fCentY + lprect_new.fHeight/2, dwImgH-1.f);

    dwX0_1 = dwX0_1 + adwMarginHW[1] - dwX0_0;
    dwY0_1 = dwY0_1 + adwMarginHW[0] - dwY0_0;
    dwX1_1 = dwX1_1 + adwMarginHW[1] - dwX0_0;
    dwY1_1 = dwY1_1 + adwMarginHW[0] - dwY0_0;

    stIIR.pfImage_0 = pfBlkBuffer_0;
    stIIR.pfImage_1 = pfBlkBuffer_1;
    stIIR.dwH = dwBlkH;
    stIIR.dwW = dwBlkW;
    stIIR.rect.dwX0 = dwX0_1;
    stIIR.rect.dwY0 = dwY0_1;
    stIIR.rect.dwX1 = dwX1_1;
    stIIR.rect.dwY1 = dwY1_1;
  #if LPDR_DBG&0
    cv::Mat normpre(dwBlkH, dwBlkW, CV_32FC1, pfBlkBuffer_0);
    cv::imshow("normpre", normpre);
  #endif
    doNormContrastBB_f(pfBlkBuffer_0, dwBlkH, dwBlkW, stIIR.rect);
  #if LPDR_DBG&0
    cv::Mat normafter(dwBlkH, dwBlkW, CV_32FC1, pfBlkBuffer_0);
    cv::imshow("normafter", normafter);
    cv::waitKey(0);
  #endif
    memcpy(pfBlkBuffer_1, pfBlkBuffer_0, sizeof(float) * dwBlkH * dwBlkW);
    
//    printf(">>>wwww:[%d/%d, %d]\n", dwNowMissionID, dwMissionNum, dwThreadID);
    
    int dwRet = doRecogOne(hPREG, hCHRECOG, &stIIR, &stOut);
    
//    printf(">>>dwRet:%d, [%d/%d, %d]\n", dwRet, dwNowMissionID, dwMissionNum, dwThreadID);
    if (!dwRet)
    {
      //write data
      unique_lock<mutex> countlc2(*p_missionmt);
//      p_missionmt->lock();
      
      stOut.adwLPRect[0] = stIIR.rect.dwX0 + dwX0_0 - adwMarginHW[1];
      stOut.adwLPRect[1] = stIIR.rect.dwY0 + dwY0_0 - adwMarginHW[0];
      stOut.adwLPRect[2] = stIIR.rect.dwX1 + dwX0_0 - adwMarginHW[1];
      stOut.adwLPRect[3] = stIIR.rect.dwY1 + dwY0_0 - adwMarginHW[0];
      pstLPDRSetOne->astLPs[pstLPDRSetOne->dwLPNum++] = stOut;
      
//      printf(">>>>dwLPNum:%x, %d[%d/%d]\n", &pstLPDRSetOne->dwLPNum, pstLPDRSetOne->dwLPNum, dwNowMissionID, dwThreadID);
      
      countlc2.unlock();
//      p_missionmt->unlock();
    }
  }
  
  delete []pbyBuffer;
  delete []pfBlkBuffer_0;
  delete []pfBlkBuffer_1;

//  printf("finished thread:%d\n", dwThreadID);

  int *p_dwFinishCount = pstGlobal->p_dwFinishCount;
  int dwNeedFinishNum = pstGlobal->dwNeedFinishNum;
  int dwFinishCount = 0;
  
  unique_lock<mutex> countlc(*p_countmt);
  if ((*p_dwFinishCount) < dwNeedFinishNum) {
    (*p_dwFinishCount)++;
  }
  dwFinishCount = (*p_dwFinishCount);
//  printf("fucking finished thread_0:%d, %d/%d\n", dwThreadID, *p_dwFinishCount, dwNeedFinishNum);
  if (dwFinishCount == dwNeedFinishNum) {
    p_cv->notify_all();
  }

  countlc.unlock();
//  printf("fucking finished thread_1:%d, %d/%d\n", dwThreadID, *p_dwFinishCount, dwNeedFinishNum);
  
  
  return 0;
}

#endif



