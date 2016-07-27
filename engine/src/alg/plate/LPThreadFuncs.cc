
#include "LPThreadFuncs.hpp"



int doRecogOne(LPDR_HANDLE hPolyReg, LPDR_HANDLE hChRecog, InputInfoRecog_S *pstIIR, LPDRInfo_S *pstOut);


#if DO_FCN_THREAD

struct LP_RFCND_THREAD_S {
  LPDR_ImageInner_S *pstImage;
  float *pfStdInputData;
  int dwStdH;
  int dwStdW;
  int *pdwRealW;
  int *pdwRealH;
};


void *lpReadyFCNDataThreadOne(void *pParam);

int lpReadyFCNDataThreads(LPDR_ImageInner_S *pstImgSet, int dwImgNum, ModuleFCNN_S *pstFCNN)
{
  int dwTI;
  LP_RFCND_THREAD_S astParams[256];
  pthread_t athdIDs[256];
  assert(256>=dwImgNum);
  
  int dwStdH, dwStdW;
  
  dwStdH = pstFCNN->adwInShape[2];
  dwStdW = pstFCNN->adwInShape[3];
  for (dwTI = 0; dwTI < dwImgNum; dwTI++)
  {
    LPDR_ImageInner_S *pstImage = &pstImgSet[dwTI];
    astParams[dwTI].pfStdInputData = pstFCNN->pfInputData + dwTI * dwStdW * dwStdH;
    astParams[dwTI].pdwRealW = &pstFCNN->pdwRealWs[dwTI];
    astParams[dwTI].pdwRealH = &pstFCNN->pdwRealHs[dwTI];
    astParams[dwTI].dwStdW = dwStdW;
    astParams[dwTI].dwStdH = dwStdH;
    astParams[dwTI].pstImage = pstImage;
    
    int dwRet = pthread_create(&athdIDs[dwTI], NULL, lpReadyFCNDataThreadOne, (void*)&astParams[dwTI]);
  }
  
  for (dwTI = 0; dwTI < dwImgNum; dwTI++)
  {
    pthread_join(athdIDs[dwTI], NULL);
  }

  return 0;
}


void *lpReadyFCNDataThreadOne(void *pParam)
{
  int dwRealW, dwRealH;
  
  LP_RFCND_THREAD_S *pstParam = (LP_RFCND_THREAD_S*)pParam;
  LPDR_ImageInner_S *pstImage = pstParam->pstImage;
  float *pfStdInputData = pstParam->pfStdInputData;
  int dwStdH = pstParam->dwStdH;
  int dwStdW = pstParam->dwStdW;
  int *pdwRealH = pstParam->pdwRealH;
  int *pdwRealW = pstParam->pdwRealW;

  float *pfDataOri = pstImage->pfData;
  int dwImgWOri = pstImage->dwImgW;
  int dwImgHOri = pstImage->dwImgH;
  
  memset(pfStdInputData, 0, sizeof(float) * dwStdW * dwStdH);
  imgResizeAddBlack_f(pfDataOri, dwImgWOri, dwImgHOri, pfStdInputData, dwStdW, dwStdH, &dwRealW, &dwRealH);
  
  *pdwRealH = dwRealH;
  *pdwRealW = dwRealW;
  
  return 0;
}


struct LP_PP_THREAD_S {
  LPDR_Image_S *pstOneIn;
  LPDR_ImageInner_S *pstOne;
};

void *lpPreProcessThreadOne(void *pParam);

int lpPreProcessThreads(LPDR_ImageSet_S *pstImgSet, LPDR_ImageInner_S *pstFCNNImgSet)
{
  int dwI;
  int dwImgNum = pstImgSet->dwImageNum;
  LP_PP_THREAD_S astPPs[256];
  pthread_t athdIDs[256];
  assert(dwImgNum<=256);
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    LPDR_Image_S *pstImgIn = &pstImgSet->astSet[dwI];
    LPDR_ImageInner_S *pstOne = &pstFCNNImgSet[dwI];
    astPPs[dwI].pstOneIn = pstImgIn;
    astPPs[dwI].pstOne = pstOne;
    int dwRet = pthread_create(&athdIDs[dwI], NULL, lpPreProcessThreadOne, (void*)&astPPs[dwI]);
  }

  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    pthread_join(athdIDs[dwI], NULL);
  }
  
  return 0;
}


void *lpPreProcessThreadOne(void *pParam)
{
  LP_PP_THREAD_S *pstPP = (LP_PP_THREAD_S*)pParam;
  LPDR_Image_S *pstOneIn = pstPP->pstOneIn;
  LPDR_ImageInner_S *pstOne = pstPP->pstOne;
  
  int dwImgW = pstOneIn->dwImgW;
  int dwImgH = pstOneIn->dwImgH;
  pstOne->dwImgW = dwImgW;
  pstOne->dwImgH = dwImgH;
  pstOne->pfData = new float[pstOne->dwImgW * pstOne->dwImgH];
  int dwSize = pstOne->dwImgW * pstOne->dwImgH;
  
  cv::Mat inputColorOne(dwImgH, dwImgW, CV_8UC3, pstOneIn->pubyData);
  cv::Mat inputGrayOne(dwImgH, dwImgW, CV_8UC1);
  cv::cvtColor(inputColorOne, inputGrayOne, CV_BGR2GRAY);

  uchar *pubyOne = (uchar*)inputGrayOne.data;
  cv::Mat oneData(dwImgH, dwImgW, CV_32FC1, pstOne->pfData);
  inputGrayOne.convertTo(oneData, CV_32FC1, 1.0f/255.f, 0);  

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
  pthread_mutex_t *pmutex;

  vector<REG_RECOG_MISSION_S> *pvecMission; //lock
  int dwNowMissionID; //lock
  int dwMissionNum;
	int dwDev_Type;
	int dwDev_ID;
};

struct REG_RECOG_S {
  LPDR_HANDLE hPREG;
  LPDR_HANDLE hCHRECOG;
  
  int dwThreadID;

  REG_RECOG_GLOBAL_S *pstGlobal;
};


void *doRecogOne_Thread(void *pParams);
int doRecognitions_Threads(LPDR_HANDLE handle, LPDR_ImageInner_S *pstImgSet, int dwImgNum, LPDR_OutputSet_S *pstOutputSet)
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
    }
  }
  
  pthread_mutex_t mutex;
  
  pthread_mutex_init(&mutex, NULL);

  int dwMissionNum = vecMissions.size();
  
  REG_RECOG_S astParams[MAX_RECOG_THREAD_NUM];
  REG_RECOG_GLOBAL_S stGlobal;
  stGlobal.pvecMission = &vecMissions;
  stGlobal.pmutex = &mutex;
  stGlobal.dwNowMissionID = 0;
  stGlobal.dwMissionNum = dwMissionNum;
  stGlobal.dwDev_Type = pstLPDR->dwDev_Type;
	stGlobal.dwDev_ID = pstLPDR->dwDev_ID;
  
  pthread_t athdIDs[MAX_RECOG_THREAD_NUM];
  int dwNeedThreadNum = min(dwMissionNum, MAX_RECOG_THREAD_NUM);
//  cout << "Need Thread Number:" << dwNeedThreadNum << endl;
  for (int dwTI = 0; dwTI < dwNeedThreadNum; dwTI++)
  {
 //   printf("start new thread %d\n", dwTI);
    REG_RECOG_S *pstParam = &astParams[dwTI];
    pstParam->pstGlobal = &stGlobal;
    pstParam->hPREG = pstLPDR->ahPREGs[dwTI];
    pstParam->hCHRECOG = pstLPDR->ahCHRECOGs[dwTI];
    pstParam->dwThreadID = dwTI;

    int dwRet = pthread_create(&athdIDs[dwTI], NULL, doRecogOne_Thread, (void*)pstParam);
  }
  
  for (int dwTI = 0; dwTI < dwNeedThreadNum; dwTI++)
  {
    pthread_join(athdIDs[dwTI], NULL);
  }
  
  pthread_mutex_destroy(&mutex);
  
  pstOutputSet->dwImageNum = dwImgNum;

#if LPDR_TIME&1
  gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec) / 1000.f;
	printf("doRecognitions threads cost:%.2fms\n", diff);
#endif
  return 0;
}


void *doRecogOne_Thread(void *pParam)
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
  
  LPRectInfo lprect;

  pthread_mutex_t *pmutex = pstGlobal->pmutex;

  pfBlkBuffer_0 = new float[dwBlkMaxLen];
  pfBlkBuffer_1 = new float[dwBlkMaxLen];
  pbyBuffer = new char[dwBufferLen];
  
  while (1) {
    int dwMissionIDNow = -1;
    pthread_mutex_lock(pmutex);
    dwMissionIDNow = pstGlobal->dwNowMissionID;
    pthread_mutex_unlock(pmutex);
    if (dwMissionIDNow >= dwMissionNum)
    {
//      cout << "no more mission now!" << endl;
      break;
    }
    

    //read data
    pthread_mutex_lock(pmutex);
    
//    printf("start new mission %x %d/%d[%d]\n", &pstGlobal->dwNowMissionID, pstGlobal->dwNowMissionID+1, dwMissionNum, dwThreadID);
    
    REG_RECOG_MISSION_S &stMission = (*pvecMission)[pstGlobal->dwNowMissionID];
    pstGlobal->dwNowMissionID++;
    
    pfImage = stMission.pfMomImage;
    dwImgW = stMission.dwImgW;
    dwImgH = stMission.dwImgH;
    lprect = stMission.stLPRect;
    pstLPDRSetOne = stMission.pstLPDRSetOne;

    pthread_mutex_unlock(pmutex);

    //process data
    LPRectInfo lprect_crop(lprect.fScore, lprect.fCentY, lprect.fCentX, lprect.fHeight*afCropSize[1], lprect.fWidth*afCropSize[0]);

    LPDRInfo_S stOut;
    InputInfoRecog_S stIIR;
    
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
    
    int dwRet = doRecogOne(hPREG, hCHRECOG, &stIIR, &stOut);
    
    if (!dwRet)
    {
      //write data
      pthread_mutex_lock(pmutex);
      
      stOut.adwLPRect[0] = stIIR.rect.dwX0 + dwX0_0 - adwMarginHW[1];
      stOut.adwLPRect[1] = stIIR.rect.dwY0 + dwY0_0 - adwMarginHW[0];
      stOut.adwLPRect[2] = stIIR.rect.dwX1 + dwX0_0 - adwMarginHW[1];
      stOut.adwLPRect[3] = stIIR.rect.dwY1 + dwY0_0 - adwMarginHW[0];
      pstLPDRSetOne->astLPs[pstLPDRSetOne->dwLPNum++] = stOut;
      
      pthread_mutex_unlock(pmutex);
    }
  }
  
  delete []pbyBuffer;
  delete []pfBlkBuffer_0;
  delete []pfBlkBuffer_1;

  return 0;
}
#endif



