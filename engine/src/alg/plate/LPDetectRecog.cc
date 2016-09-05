
#include "LPFCNN.hpp"
#include "LPRPN.hpp"
#include "LPROIP.hpp"
#include "LPPREG.hpp"
#include "LPCHRECOG.hpp"
#include "LPThreadFuncs.hpp"
#include "LPThreadFuncsQueue.hpp"
#include "LPCOLOR.hpp"

#define LPDR_CLASS_NUM 79

const char *paInv_chardict[LPDR_CLASS_NUM] = {"_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
          "A", "B", "C", "D", "E", "F", "G", "H", "J", \
          "K", "L", "M", "N", "P", "Q", "R", "S", "T",\
          "U", "V", "W", "X", "Y", "Z", "I", "京", "津",\
          "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘", \
          "皖", "闽", "鲁", "新", "苏", "浙", "赣", "鄂", \
          "桂", "甘", "晋", "蒙", "陕", "吉", "贵", "粤", \
          "青", "藏", "川", "宁", "琼", "使", "领", "试", \
          "学", "临", "时", "警", "港", "O", "挂", "澳", "#"};

int doRecogOne(LPDR_HANDLE hPolyReg, LPDR_HANDLE hChRecog, InputInfoRecog_S *pstIIR, LPDRInfo_S *pstOut);
int doRecognitions(LPDR_HANDLE handle, LPDR_ImageInner_S *pstImgSet, int dwImgNum, LPDR_OutputSet_S *pstOutputSet);
int doRecogColors(LPDR_HANDLE handle, LPDR_ImageSet_S *pstImgSet, LPDR_OutputSet_S *pstOutputSet);
int doRecogColors_NN(LPDR_HANDLE hCOLOR, LPDR_ImageSet_S *pstImgSet, LPDR_OutputSet_S *pstOutputSet);

int LPDR_Create(LPDR_HANDLE *pHandle, LPDRConfig_S *pstConfig)
{
    LPDR_Info_S *pstLPDR = (LPDR_Info_S*)calloc(1, sizeof(LPDR_Info_S));
    *pHandle = (LPDR_HANDLE)pstLPDR;
    int dwDevType = pstConfig->dwDevType;
    int dwDevID = pstConfig->dwDevID;
    pstLPDR->dwDev_Type = dwDevType;
    pstLPDR->dwDev_ID = dwDevID;
    
    int dwGroupSize = pstConfig->stFCNN.adwShape[0];
    pstLPDR->pvBBGroupOfROIP = new vector<LPRectInfo>[dwGroupSize];
    pstLPDR->pvBBGroupOfNMS = new vector<LPRectInfo>[dwGroupSize];

#if DO_FCN_THREAD
    pstLPDR->p_ppTPool = new dg::ThreadPool(dwGroupSize);
    pstLPDR->p_rfcnTPool = new dg::ThreadPool(dwGroupSize);
#endif

//    cout << "cat 0\n";
    LPFCNN_Create(pstConfig->stFCNN, dwDevType, dwDevID, &pstLPDR->hFCNN);
    
//    cout << "cat 1\n";
    LPRPN_Create(pstConfig->stRPN, dwDevType, dwDevID, &pstLPDR->hRPN);
    
//    cout << "cat 2\n";
#if 1
    ModuleRPN_S *pstRPN = (ModuleRPN_S*)(pstLPDR->hRPN);
    int *pdwRPNOutFeatShape = pstRPN->adwOutShape + 3;
    int *pdwROIPInShape = pstConfig->stROIP.adwShape;
    for (int i = 0; i < 4; i++)
    {
      pdwROIPInShape[i] = pdwRPNOutFeatShape[i];
//      cout << pdwRPNOutFeatShape[i] << endl;
    }
#else
    ModuleRPN_S *pstRPN = (ModuleRPN_S*)(pstLPDR->hRPN);
    int *pdwRPNOutFeatShape = pstRPN->adwOutShape + 8;
    int *pdwROIPInShape = pstConfig->stROIP.adwShape;
    for (int i = 0; i < 4; i++)
    {
      pdwROIPInShape[i] = pdwRPNOutFeatShape[i];
//      cout << pdwRPNOutFeatShape[i] << endl;
    }
#endif
//    cout << "cat 3\n";
    LPROIP_Create(pstConfig->stROIP, dwDevType, dwDevID, &pstLPDR->hROIP);
    
//    cout << "cat 4\n";
#if MAX_RECOG_THREAD_NUM>1
    
    for (int dwTI = 0; dwTI < MAX_RECOG_THREAD_NUM; dwTI++)
    {
      LPPREG_Create(pstConfig->stPREG, dwDevType, dwDevID, &pstLPDR->ahPREGs[dwTI]);
      LPCHRECOG_Create(pstConfig->stCHRECOG, dwDevType, dwDevID, &pstLPDR->ahCHRECOGs[dwTI]);
    }
    
    pstLPDR->p_recogTPool = new dg::ThreadPool(MAX_RECOG_THREAD_NUM);
#else
    LPPREG_Create(pstConfig->stPREG, dwDevType, dwDevID, &pstLPDR->hPREG);
    LPCHRECOG_Create(pstConfig->stCHRECOG, dwDevType, dwDevID, &pstLPDR->hCHRECOG);
#endif

    LPCOLOR_Create(pstConfig->stCOLOR, dwDevType, dwDevID, &pstLPDR->hCOLOR);

    return 0;
}



bool doSortRect(LPRectInfo one, LPRectInfo two);


int LPDR_Process(LPDR_HANDLE handle, LPDR_ImageSet_S *pstImgSet, LPDR_OutputSet_S *pstOutputSet)
{
  LPDR_Info_S *pstLPDR = (LPDR_Info_S*)handle;
  LPDR_HANDLE hFCNN = pstLPDR->hFCNN;
  int dwI, dwJ, dwRI, dwPI;
  int dwImgNum = pstImgSet->dwImageNum;
  
//  assert(dwImgNum==1);
  
  ////////////////FCN///////////////
#if LPDR_TIME  
  float costtime, diff;
  struct timeval start, end;
#endif

#if LPDR_TIME
  gettimeofday(&start, NULL);
#endif

  LPDR_ImageInner_S *pstFCNNImgSet = new LPDR_ImageInner_S[dwImgNum];
  memset(pstFCNNImgSet, 0, sizeof(LPDR_ImageInner_S) * dwImgNum);
#if DO_FCN_THREAD //dog
//  lpPreProcessThreads(pstImgSet, pstFCNNImgSet);
  lpPreProcessThreadsQueue(pstLPDR->p_ppTPool, pstImgSet, pstFCNNImgSet);
#else
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    LPDR_ImageInner_S *pstOne = &pstFCNNImgSet[dwI];
    int dwImgW = pstImgSet->astSet[dwI].dwImgW;
    int dwImgH = pstImgSet->astSet[dwI].dwImgH;
    pstOne->dwImgW = dwImgW;
    pstOne->dwImgH = dwImgH;
    pstOne->pfData = new float[pstOne->dwImgW * pstOne->dwImgH];
    int dwSize = pstOne->dwImgW * pstOne->dwImgH;
    memset(pstOne->pfData, 0, sizeof(float) * dwSize);
    
    cv::Mat inputColorOne(dwImgH, dwImgW, CV_8UC3, pstImgSet->astSet[dwI].pubyData);
    cv::Mat inputGrayOne(dwImgH, dwImgW, CV_8UC1);
    cv::cvtColor(inputColorOne, inputGrayOne, CV_BGR2GRAY);

    uchar *pubyOne = (uchar*)inputGrayOne.data;
    cv::Mat oneData(dwImgH, dwImgW, CV_32FC1, pstOne->pfData);
    inputGrayOne.convertTo(oneData, CV_32FC1, 1.0f/255.f, 0);

//    doRotate_f(pstOne->pfData, dwImgW, dwImgH, 10.0f);

#if LPDR_DBG&1
  cv::Mat imagegray(dwImgH, dwImgW, CV_32FC1, pstOne->pfData);
  cv::namedWindow("fuck0", 0);
  cv::imshow("fuck0", imagegray);
  cv::waitKey(0);
#endif
  }
#endif //dog
#if LPDR_TIME
  gettimeofday(&end, NULL);
  diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
  printf("pre fcnn cost:%.2fms\n", diff);
#endif

  LPFCNN_Process(hFCNN, pstFCNNImgSet, dwImgNum, pstLPDR->p_rfcnTPool);
  
  ////////////////RPN///////////////
#if 1
#if LPDR_TIME
  gettimeofday(&start, NULL);
#endif
  LPDR_HANDLE hRPN = pstLPDR->hRPN;
  ModuleFCNN_S *pstFCNN = (ModuleFCNN_S*)hFCNN;
  
  ModuleRPN_S *pstRPN = (ModuleRPN_S*)hRPN;
  int dwRPNWantedAll = pstRPN->adwInShape[0] * pstRPN->adwInShape[1];
  int dwRPNWantedPerImg = pstRPN->adwInShape[1];
  LPDR_ImageInner_S *pstRPNImgSet = new LPDR_ImageInner_S[dwRPNWantedAll];
  memset(pstRPNImgSet, 0, sizeof(LPDR_ImageInner_S) * dwRPNWantedAll);
//  cout << "fuck rpn 0\n";
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    LPDR_ImageInner_S *pstRPNImgNowSet = pstRPNImgSet + dwI * dwRPNWantedPerImg;
    LPDR_ImageInner_S *pstOne = &pstFCNNImgSet[dwI];
    int dwImgWSrc = pstOne->dwImgW;
    int dwImgHSrc = pstOne->dwImgH;
    float *pfDataSrc = pstOne->pfData;
//    cout << dwI << "_pfDataSrc:" << pfDataSrc << endl;
    if (!pfDataSrc) continue;
    vector<LPRectInfo> &lpgroup = pstFCNN->plpgroup[dwI];
    
    int dwBBNum = lpgroup.size();
//    cout << "dwBBNum:" << dwBBNum << endl;
    int dwActualNUm = dwBBNum < dwRPNWantedPerImg ? dwBBNum : dwRPNWantedPerImg;
    
    int dwRI2 = 0;
    for (dwJ = 0; dwJ < dwActualNUm; dwJ++)
    {
      LPDR_ImageInner_S *pstRPNImgNow = pstRPNImgNowSet + dwJ;
      
      LPRectInfo *prect = &lpgroup[dwJ];
      int dwX0 = prect->fCentX - prect->fWidth/2;
      int dwY0 = prect->fCentY - prect->fHeight/2;
      int dwX1 = prect->fCentX + prect->fWidth/2;
      int dwY1 = prect->fCentY + prect->fHeight/2;
      
      pstRPNImgNow->dwImgW = dwX1 - dwX0;
      pstRPNImgNow->dwImgH = dwY1 - dwY0;
//      cout << pstRPNImgNow->dwImgW << ", " <<  dwX1-dwX0 << ", " << pstRPNImgNow->dwImgH << ", " << dwY1-dwY0 << endl;
//      assert(pstRPNImgNow->dwImgW==dwX1-dwX0&&pstRPNImgNow->dwImgH==dwY1-dwY0);
      int dwSize = pstRPNImgNow->dwImgW * pstRPNImgNow->dwImgH;
//      cout << "dwSize:" << dwSize << ", pstRPNImgNow->dwImgW:" << pstRPNImgNow->dwImgW << ", " << "pstRPNImgNow->dwImgH:" << pstRPNImgNow->dwImgH << endl;
      pstRPNImgNow->pfData = new float[dwSize];
      pstRPNImgNow->dwPID = dwI;
      pstRPNImgNow->adwPRect[0] = dwX0;
      pstRPNImgNow->adwPRect[1] = dwY0;
      pstRPNImgNow->adwPRect[2] = dwX1;
      pstRPNImgNow->adwPRect[3] = dwY1;
      memset(pstRPNImgNow->pfData, 0, sizeof(float) * dwSize);
      for (dwRI = dwY0, dwRI2 = 0; dwRI < dwY1; dwRI++, dwRI2++)
      {
        memcpy(pstRPNImgNow->pfData + dwRI2 * pstRPNImgNow->dwImgW, pfDataSrc + dwRI * dwImgWSrc + dwX0, sizeof(float) * pstRPNImgNow->dwImgW);
      }
    }
  }
#if LPDR_TIME
  gettimeofday(&end, NULL);
  diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
  printf("pre rpn cost:%.2fms\n", diff);
#endif
//  cout << "fuck rpn 1\n";
  LPRPN_Process(hRPN, pstRPNImgSet, dwRPNWantedAll);
//  cout << "fuck rpn 2\n";
#if LPDR_DBG
  vector<cv::Mat> avCImages;
  vector<string> avWinNames;
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    int dwImgW = pstImgSet->astSet[dwI].dwImgW;
    int dwImgH = pstImgSet->astSet[dwI].dwImgH;
    cv::Mat cimg(dwImgH, dwImgW, CV_8UC3, pstImgSet->astSet[dwI].pubyData);
    
    avCImages.push_back(cimg);
    
    stringstream ss;
    ss << "rpnout_" << dwI;
    string winame = ss.str();
    
    avWinNames.push_back(winame);
  }
#endif

#if LPDR_DBG
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    cv::Mat cimg = avCImages[dwI];
    for (dwJ = 0; dwJ < dwRPNWantedPerImg; dwJ++)
    {
      vector<LPRectInfo> &rpnrects = pstRPN->plprectgroup[dwI * dwRPNWantedPerImg + dwJ];

      for (dwRI = 0; dwRI < rpnrects.size(); dwRI++)
      {
        LPRectInfo *prect = &rpnrects[dwRI];

        int dwX0 = prect->fCentX - prect->fWidth/2;
        int dwY0 = prect->fCentY - prect->fHeight/2;
        int dwX1 = prect->fCentX + prect->fWidth/2;
        int dwY1 = prect->fCentY + prect->fHeight/2;
        
        cv::rectangle(cimg, cv::Point(dwX0, dwY0), cv::Point(dwX1, dwY1), CV_RGB(255, 0, 0), 2, 8, 0);
      }
    }
    string winame = avWinNames[dwI];
    cv::namedWindow(winame, 0);
    cv::imshow(winame, cimg);
  }
  cv::waitKey(10);
#endif
#endif
//  cout << "fuck rpn 3\n";
  ///////////////ROIP////////////////
#if 1
//  cout << "fuck roip 0\n";
  LPDR_HANDLE hROIP = pstLPDR->hROIP;
  ModuleROIP_S *pstROIP = (ModuleROIP_S*)hROIP;
  int *pdwRectSZ = pstROIP->adwRectSZ;
  float *pfRect3D = pstROIP->pfRect3D;
//  cout << "roip:" << pdwRectSZ[0] * pdwRectSZ[1] * pdwRectSZ[2] << endl;
  memset(pfRect3D, 0, sizeof(float) * pdwRectSZ[0] * pdwRectSZ[1] * pdwRectSZ[2]);
//  cout << "fuck roip 1\n";
  int dwGroupNum = pstRPN->dwGroupSize;
  assert(pdwRectSZ[1]==pstROIP->adwInShape[4+1]);
  int dwRectWantedPerRPNImg = pdwRectSZ[1];
  int dwROIPBatchNum = 0, dwROIPBatchNowID;
//  cout << dwGroupNum << ", " << dwImgNum << ", " << dwRPNWantedPerImg << endl;
//  assert(dwGroupNum==dwImgNum*dwRPNWantedPerImg);
  int adwIndex[1024];
  vector<int> *prectidxgroup = new vector<int>[dwGroupNum];
//  cout << "fuck roip 2\n";
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    for (dwJ = 0; dwJ < dwRPNWantedPerImg; dwJ++)
    {
      dwROIPBatchNum++;
      dwROIPBatchNowID = dwROIPBatchNum - 1;
      vector<int> &rectidxgroup = prectidxgroup[dwI * dwRPNWantedPerImg + dwJ];
      vector<LPRectInfo> &lprectgroup = pstRPN->plprectgroup_0[dwI * dwRPNWantedPerImg + dwJ];
      vector<LPRectInfo> &lprectgroup2 = pstRPN->plprectgroup[dwI * dwRPNWantedPerImg + dwJ];
      pfRect3D = pstROIP->pfRect3D + dwROIPBatchNowID * pdwRectSZ[1] * pdwRectSZ[2];
      int dwRectnum = lprectgroup.size();
      if (dwRectnum == 0) continue;
//      printf("img:%d, roi:%d, rectnum:%d\n", dwI, dwJ, dwRectnum);
      int dwNeedRectNum = 1024 < dwRectnum ? 1024 : dwRectnum;
      for (dwPI = 0; dwPI < dwNeedRectNum; dwPI++)
      {
        adwIndex[dwPI] = dwPI;
      }
//      random_shuffle(adwIndex, adwIndex+dwNeedRectNum);
      sort(lprectgroup.begin(), lprectgroup.end(), doSortRect);
      sort(lprectgroup2.begin(), lprectgroup2.end(), doSortRect);
      int dwNeedNum = dwRectWantedPerRPNImg < dwRectnum ? dwRectWantedPerRPNImg : dwRectnum;
//      cout << "dwRectnum:" << dwRectnum << ", dwNeedNum:" << dwNeedNum << "; ";
      for (dwRI = 0; dwRI < dwNeedNum; dwRI++)
      {
        int dwNowIdx = adwIndex[dwRI];
//        cout << dwNowIdx << ",";
        
        rectidxgroup.push_back(dwNowIdx);
        
        LPRectInfo &lprect = lprectgroup[dwNowIdx];
//        printf("%.6f,", lprect.fScore);
        
        float *pfRectOne = pfRect3D + dwRI * pdwRectSZ[2];
        pfRectOne[0] = dwROIPBatchNowID;
        pfRectOne[1] = lprect.fCentX - lprect.fWidth/2;
        pfRectOne[2] = lprect.fCentY - lprect.fHeight/2;
        pfRectOne[3] = lprect.fCentX + lprect.fWidth/2;
        pfRectOne[4] = lprect.fCentY + lprect.fHeight/2;
      }
//      cout << endl;
      
    }
  }
//  cout << "fuck roip 3\n";
//  pstRPN->pdwRealWs;
//  pstRPN->pdwRealHs;
  LPROIP_Process(hROIP, pstRPN->pfOutputDataFeat, pstRPN->adwOutShape+3, pstROIP->pfRect3D, pstROIP->adwRectSZ);
  float *pfOutCls = pstROIP->pfOutCls;
//  cout << "fuck roip 4\n";
  
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    vector<LPRectInfo> &lproipgroup_one = pstLPDR->pvBBGroupOfROIP[dwI];
    lproipgroup_one.clear();
    for (dwJ = 0; dwJ < dwRPNWantedPerImg; dwJ++)
    {
      float *pfCls = pfOutCls + (dwI * dwRPNWantedPerImg + dwJ) * dwRectWantedPerRPNImg;
      vector<int> &rectidxgroup = prectidxgroup[dwI * dwRPNWantedPerImg + dwJ];
      vector<LPRectInfo> &lprectgroup = pstRPN->plprectgroup[dwI * dwRPNWantedPerImg + dwJ];

//      cout << rectidxgroup.size() << ", " << lprectgroup.size() << endl;

      assert(rectidxgroup.size()<=lprectgroup.size());
      int dwRectnum = rectidxgroup.size();
      if (dwRectnum == 0) continue;
      int dwNeedNum = dwRectWantedPerRPNImg < dwRectnum ? dwRectWantedPerRPNImg : dwRectnum;
      for (dwRI = 0; dwRI < dwNeedNum; dwRI++)
      {
        int dwNowIdx = rectidxgroup[dwRI];
        LPRectInfo &lprect = lprectgroup[dwNowIdx];
//        printf("%.6f, ", lprect.fScore);
        
//        cout << pfCls[dwRI] << ", ";
        if (pfCls[dwRI] > LP_ROIP_SCORE_MAX)
        {
          lproipgroup_one.push_back(lprect);
#if LPDR_DBG&1
          cv::Mat cimg = avCImages[dwI];
          
          int dwX0 = lprect.fCentX - lprect.fWidth/2;
          int dwY0 = lprect.fCentY - lprect.fHeight/2;
          int dwX1 = lprect.fCentX + lprect.fWidth/2;
          int dwY1 = lprect.fCentY + lprect.fHeight/2;
          cv::rectangle(cimg, cv::Point(dwX0, dwY0), cv::Point(dwX1, dwY1), CV_RGB(255, 255, 0), 2, 8, 0);
#endif
        }
      }
//      printf("\n");
    }
  }
//  cout << "fuck roip 6\n";

#endif
#if LPDR_TIME
  gettimeofday(&start, NULL);
#endif
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    vector<LPRectInfo> &lproipgroup_one = pstLPDR->pvBBGroupOfROIP[dwI];
    vector<LPRectInfo> &lproipnms_one = pstLPDR->pvBBGroupOfNMS[dwI];
    group_bbs(lproipgroup_one, lproipnms_one, 0.3);
    
#if LPDR_DBG&1
    cv::Mat cimg = avCImages[dwI];
    
    for (dwRI = 0; dwRI < lproipnms_one.size(); dwRI++)
    {
      LPRectInfo &lprect = lproipnms_one[dwRI];
      
      int dwX0 = lprect.fCentX - lprect.fWidth/2;
      int dwY0 = lprect.fCentY - lprect.fHeight/2;
      int dwX1 = lprect.fCentX + lprect.fWidth/2;
      int dwY1 = lprect.fCentY + lprect.fHeight/2;
      
      cv::rectangle(cimg, cv::Point(dwX0, dwY0), cv::Point(dwX1, dwY1), CV_RGB(0, 255, 0), 3, 8, 0);
    }
#endif
  }
#if LPDR_TIME
  gettimeofday(&end, NULL);
  diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
  printf("nms cost:%.2fms\n", diff);
#endif
#if LPDR_DBG
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    cv::Mat cimg = avCImages[dwI];
    string winame = avWinNames[dwI];
    cv::namedWindow(winame, 0);
    cv::imshow(winame, cimg);
  }
  cv::waitKey(10);
#endif

  ///////////////////////////////
#if MAX_RECOG_THREAD_NUM>1
//  doRecognitions_Threads(handle, pstFCNNImgSet, dwImgNum, pstOutputSet);
  doRecognitions_ThreadsQueue(pstLPDR->p_recogTPool, handle, pstFCNNImgSet, dwImgNum, pstOutputSet);
#else
  doRecognitions(handle, pstFCNNImgSet, dwImgNum, pstOutputSet);
#endif
//  doRecogColors(handle, pstImgSet, pstOutputSet);

  doRecogColors_NN(pstLPDR->hCOLOR, pstImgSet, pstOutputSet);

  ///////////////////////////////
  //release resources
#if 1
//  cout << "fuck release 0\n";
  for (dwI = 0; dwI < dwRPNWantedAll; dwI++)
  {
    LPDR_ImageInner_S *pstOne = &pstRPNImgSet[dwI];
    if (pstOne->pfData)
    {
      delete []pstOne->pfData;
      pstOne->pfData = 0;
    }
  }
//  cout << "fuck release 1\n";
  delete []pstRPNImgSet;
  pstRPNImgSet = 0;
#endif

//  cout << "fuck release 2\n";
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    LPDR_ImageInner_S *pstOne = &pstFCNNImgSet[dwI];
    delete []pstOne->pfData;
    pstOne->pfData = 0;
  }
//  cout << "fuck release 3\n";
  delete []pstFCNNImgSet;
  pstFCNNImgSet = 0;
//  cout << "fuck release 4\n";
  
  return 0;
}


int LPDR_Release(LPDR_HANDLE handle)
{
    LPDR_Info_S *pstLPDR = (LPDR_Info_S*)handle;

#if DO_FCN_THREAD
    delete pstLPDR->p_ppTPool;
    delete pstLPDR->p_rfcnTPool;
#endif
    
#if MAX_RECOG_THREAD_NUM>1
    delete pstLPDR->p_recogTPool;
#endif    
//    cout << "release 0\n";
    LPFCNN_Release(pstLPDR->hFCNN);
    
//    cout << "release 1\n";
    LPRPN_Release(pstLPDR->hRPN);
    
//    cout << "release 2\n";
    LPROIP_Release(pstLPDR->hROIP);
    
//    cout << "release 3\n";
#if MAX_RECOG_THREAD_NUM>1
    
    for (int dwTI = 0; dwTI < MAX_RECOG_THREAD_NUM; dwTI++)
    {
      LPPREG_Release(pstLPDR->ahPREGs[dwTI]);
    }
    
//    cout << "release 4\n";
    for (int dwTI = 0; dwTI < MAX_RECOG_THREAD_NUM; dwTI++)
    {
      LPCHRECOG_Release(pstLPDR->ahCHRECOGs[dwTI]);
    }
    
#else
    LPPREG_Release(pstLPDR->hPREG);
    LPCHRECOG_Release(pstLPDR->hCHRECOG);
#endif

    LPCOLOR_Release(pstLPDR->hCOLOR);
    
    return 0;
}

bool doSortRect(LPRectInfo one, LPRectInfo two)
{
  return one.fScore > two.fScore;
}

#if MAX_RECOG_THREAD_NUM<=1
int doRecognitions(LPDR_HANDLE handle, LPDR_ImageInner_S *pstImgSet, int dwImgNum, LPDR_OutputSet_S *pstOutputSet)
{
#if LPDR_TIME&1
  float costtime, diff;
  struct timeval start, end;
  
  gettimeofday(&start, NULL);
#endif
  LPDR_Info_S *pstLPDR = (LPDR_Info_S*)handle;
  int dwI, dwJ, dwRI, dwLPI;
  int dwX0_0, dwX1_0, dwY0_0, dwY1_0, dwW_0, dwH_0;
  int dwX0_1, dwX1_1, dwY0_1, dwY1_1;
  int adwMarginHW[2];
  float afMarginRatioHW[2] = {0.4f, 0.4f};
  float afCropSize[2] = {1.6f, 2.0f};
  float afNewSize[2] = {1.2f, 1.2f};
  float *pfBlkBuffer_0 = 0, *pfBlkBuffer_1 = 0;
  char *pbyBuffer = 0;
  int dwBlkMaxLen = 1000 * 1000, dwBlkH, dwBlkW, dwBufferLen = 1000 * 1000 * 4;
  InputInfoRecog_S stIIR;
  LPDRInfo_S stOut;
  
  vector<LPRectInfo> *plproipnms = pstLPDR->pvBBGroupOfNMS;
  
  pfBlkBuffer_0 = new float[dwBlkMaxLen];
  pfBlkBuffer_1 = new float[dwBlkMaxLen];
  pbyBuffer = new char[dwBufferLen];

  stIIR.pbyBuffer = pbyBuffer;
  stIIR.dwBufferLen = dwBufferLen;
  
//  cout << "doRecognitions:0\n";
  memset(pstOutputSet, 0, sizeof(LPDR_OutputSet_S));
//  cout << "doRecognitions:1\n";
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
    dwLPI = 0;
    for (dwJ = 0; dwJ < dwSize; dwJ++)
    {
#if LPDR_DBG
      cout << "--------------------\n";
#endif
//      cout << "doRecognitions:2, " << dwSize << endl;
      LPRectInfo &lprect = lproipnms_one[dwJ];
//      cout << "H:" << lprect.fHeight << ", W:" << lprect.fWidth << endl;
      LPRectInfo lprect_crop(lprect.fScore, lprect.fCentY, lprect.fCentX, lprect.fHeight*afCropSize[1], lprect.fWidth*afCropSize[0]);

      dwX0_0 = max(lprect_crop.fCentX - lprect_crop.fWidth/2, 0.0f);
      dwY0_0 = max(lprect_crop.fCentY - lprect_crop.fHeight/2, 0.0f);
      dwX1_0 = min(lprect_crop.fCentX + lprect_crop.fWidth/2, dwImgW-1.f);
      dwY1_0 = min(lprect_crop.fCentY + lprect_crop.fHeight/2, dwImgH-1.f);
      dwW_0 = dwX1_0 - dwX0_0 + 1;
      dwH_0 = dwY1_0 - dwY0_0 + 1;
      
//      cout << "H:" << dwH_0 << ", W:" << dwW_0 << endl;

      //add margin
      adwMarginHW[0] = (int)((dwH_0 * afMarginRatioHW[0])/2);
      adwMarginHW[1] = (int)((dwW_0 * afMarginRatioHW[1])/2);
      
//      cout << adwMarginHW[0] << ", " << adwMarginHW[1] << endl;

      dwBlkH = adwMarginHW[0] * 2 + dwH_0;
      dwBlkW = adwMarginHW[1] * 2 + dwW_0;

//      assert(dwBlkH * dwBlkW < dwBlkMaxLen);
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
#endif
      memcpy(pfBlkBuffer_1, pfBlkBuffer_0, sizeof(float) * dwBlkH * dwBlkW);
      
      int dwRet = doRecogOne(pstLPDR->hPREG, pstLPDR->hCHRECOG, &stIIR, &stOut);
      if (!dwRet) 
      {
        stOut.adwLPRect[0] = stIIR.rect.dwX0 + dwX0_0 - adwMarginHW[1];
        stOut.adwLPRect[1] = stIIR.rect.dwY0 + dwY0_0 - adwMarginHW[0];
        stOut.adwLPRect[2] = stIIR.rect.dwX1 + dwX0_0 - adwMarginHW[1];
        stOut.adwLPRect[3] = stIIR.rect.dwY1 + dwY0_0 - adwMarginHW[0];
        pstLPDRSetOne->astLPs[dwLPI++] = stOut;
      }
    }
    pstLPDRSetOne->dwLPNum = dwLPI;
  }
  pstOutputSet->dwImageNum = dwImgNum;
  
  delete []pbyBuffer;
  delete []pfBlkBuffer_0;
  delete []pfBlkBuffer_1;
#if LPDR_TIME&1
  gettimeofday(&end, NULL);
  diff = ((end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec) / 1000.f;
  printf("doRecognitions cost:%.2fms\n", diff);
#endif
  return 0;
}
#endif


int mainlandLPCheck(LPDRInfo_S *pstOut);
int doRecogOneRow(LPDR_HANDLE hChRecog, LPDR_ImageInner_S *pstImage, LPRect rect, float fStrechRatio, float fShrinkRatio, int dwStep, float fThreshold, LPDRInfo_S *pstOut);
int parseRecogOutInfo(int *pdwClassIdx, float *pfClassScore, int dwNum, float fThreshold, LPDRInfo_S *pstOut);
int doRectifyWithPolyReg(LPDR_HANDLE hPolyReg, InputInfoRecog_S *pstIIR, int adwMRatioXY[2], float fAngle_old, float *pfAngle_new);
int doRecogOne(LPDR_HANDLE hPolyReg, LPDR_HANDLE hChRecog, InputInfoRecog_S *pstIIR, LPDRInfo_S *pstOut)
{
  int dwRet = 0;
  int dwI, dwJ;
  LPRect rect_old, rect_new;
  float fAngle_old = 0.0f, fAngle_new = 0.0f;
  int adwMRatioXY[2] = {6, 6};

#if LPDR_DBG
  {
    cv::Mat gimg(pstIIR->dwH, pstIIR->dwW, CV_32FC1, pstIIR->pfImage_0);
    cv::Mat cimg(pstIIR->dwH, pstIIR->dwW, CV_32FC3);
    cv::cvtColor(gimg, cimg, CV_GRAY2BGR);
    int dwX0 = pstIIR->rect.dwX0;
    int dwY0 = pstIIR->rect.dwY0;
    int dwX1 = pstIIR->rect.dwX1;
    int dwY1 = pstIIR->rect.dwY1;
    cv::rectangle(cimg, cv::Point(dwX0, dwY0), cv::Point(dwX1, dwY1), CV_RGB(255, 0, 0), 1, 8, 0);
    
    cv::imshow("rectify_pre", cimg);
    cv::waitKey(10);
  }
#endif
  
  pstIIR->dwSepY = 0;
  
  rect_old = pstIIR->rect;
  for (dwI = 0; dwI < 2; dwI++)
  {
//    cout << dwI << endl;
    dwRet = doRectifyWithPolyReg(hPolyReg, pstIIR, adwMRatioXY, fAngle_old, &fAngle_new);
    if (dwRet) break;
    rect_new = pstIIR->rect;
    //check stop condition
    float fDiff = abs(rect_new.dwY0 - rect_old.dwY0) + abs(rect_new.dwY1 - rect_old.dwY1);
    fDiff /= 2.0f;
    if (fDiff < 2.0f) break;
    rect_old = rect_new;
    fAngle_old = fAngle_new;
  }
  if (dwRet) return dwRet;
//  return 0;
  int dwSepY = pstIIR->dwSepY;
 
#if LPDR_DBG&1
  cv::Mat gimg(pstIIR->dwH, pstIIR->dwW, CV_32FC1, pstIIR->pfImage_1);
  cv::Mat cimg(pstIIR->dwH, pstIIR->dwW, CV_32FC3);
  cv::cvtColor(gimg, cimg, CV_GRAY2BGR);
  int dwX0 = pstIIR->rect.dwX0;
  int dwY0 = pstIIR->rect.dwY0;
  int dwX1 = pstIIR->rect.dwX1;
  int dwY1 = pstIIR->rect.dwY1;
  cv::rectangle(cimg, cv::Point(dwX0, dwY0), cv::Point(dwX1, dwY1), CV_RGB(255, 0, 0), 1, 8, 0);
  cv::line(cimg, cv::Point(dwX0, dwSepY), cv::Point(dwX1, dwSepY), CV_RGB(0, 255, 0), 2, 8, 0);
  cv::imshow("rectify", cimg);
  cout << "rectify" << endl;
  cv::waitKey(0);
#endif
  
  //////////////////////////////////////////
  
  
  float fStrechRatio = 5.5;
  float fShrinkRatio = 1.0;
  int dwStep = 4;
  
  LPDR_ImageInner_S stImage;
  stImage.pfData = pstIIR->pfImage_1;
  stImage.dwImgH = pstIIR->dwH;
  stImage.dwImgW = pstIIR->dwW;
  
  vector<LPRect> rects_out;
  LPRect rectUp, rectDwn;
  
  int dwYAdd0 = 0;//max(abs(rect_new.dwY1 - rect_new.dwY0 + 1)/8, 1);
  rect_new.dwY1 += dwYAdd0;
  
  if (abs(dwSepY - rect_new.dwY0 + 1) * 4 > abs(rect_new.dwY1 - rect_new.dwY0 + 1))
  {
    int dwYAdd = 0;//max(abs(dwSepY - rect_new.dwY0 + 1)/8, 1);
    rectUp.dwX0 = rect_new.dwX0; rectUp.dwY0 = rect_new.dwY0;
    rectUp.dwX1 = rect_new.dwX1; rectUp.dwY1 = dwSepY + dwYAdd;
    
    rectDwn.dwX0 = rect_new.dwX0; rectDwn.dwY0 = dwSepY;
    rectDwn.dwX1 = rect_new.dwX1; rectDwn.dwY1 = rect_new.dwY1;
    
    rects_out.push_back(rectUp);
    rects_out.push_back(rectDwn);
  }
  else
  {
    rects_out.push_back(rect_new);
  }

  memset(pstOut, 0, sizeof(LPDRInfo_S));
  if (rects_out.size()==1)
  {
    LPDRInfo_S stOut;
    
    fStrechRatio = 5.5;
    fShrinkRatio = 1.0;
    dwStep = 4;
    dwRet = doRecogOneRow(hChRecog, &stImage, rects_out[0], fStrechRatio, fShrinkRatio, dwStep, 6.0f, &stOut);
    
    *pstOut = stOut;
    pstOut->dwType = LP_TYPE_SINGLE;
  }
  else if (rects_out.size()==2)
  {
    LPDRInfo_S stOut0, stOut1;
    
    fStrechRatio = 5.5;
    fShrinkRatio = 0.8;
    dwStep = 4;
    dwRet = doRecogOneRow(hChRecog, &stImage, rects_out[0], fStrechRatio, fShrinkRatio, dwStep, 2.0f, &stOut0);
    
//    if (!dwRet)
    {
      fStrechRatio = 5.5;
      fShrinkRatio = 1.0;
      dwStep = 4;
      dwRet = doRecogOneRow(hChRecog, &stImage, rects_out[1], fStrechRatio, fShrinkRatio, dwStep, 6.0f, &stOut1);
      
      if ((stOut0.dwLPLen + stOut1.dwLPLen) > MAX_LPCHAR_NUM)
      {
        dwRet = -1;
      }
      
      if (!dwRet)
      {
        pstOut->dwLPLen = 0;
        pstOut->fAllScore = 0.f;
    //    cout << stOut0.dwLPLen << endl;
        for (dwI = 0; dwI < stOut0.dwLPLen; dwI++)
        {
          pstOut->adwLPNumber[pstOut->dwLPLen] = stOut0.adwLPNumber[dwI];
          pstOut->afScores[pstOut->dwLPLen] = stOut0.afScores[dwI];
          pstOut->dwLPLen++;
        }
        pstOut->fAllScore += stOut0.fAllScore;
        
        for (dwI = 0; dwI < stOut1.dwLPLen; dwI++)
        {
          pstOut->adwLPNumber[pstOut->dwLPLen] = stOut1.adwLPNumber[dwI];
          pstOut->afScores[pstOut->dwLPLen] = stOut1.afScores[dwI];
          pstOut->dwLPLen++;
        }
        pstOut->fAllScore += stOut1.fAllScore;
        
        pstOut->dwType = LP_TYPE_DOUBLE;
      }
    }
  }
  
  dwRet = mainlandLPCheck(pstOut);

  return dwRet;
}


int mainlandLPCheck(LPDRInfo_S *pstOut)
{
  int dwI;
  int dwNowV0, dwNowV1;
  
  //I->1, O->0
  for (dwI = 0; dwI < pstOut->dwLPLen; dwI++)
  {
    int dwNowV = pstOut->adwLPNumber[dwI];
    int dwNewV = dwNowV;
    if (dwNowV == 35) //I->1
    {
      dwNewV = 2;
    }
    else if (dwNowV == 75) //O->0
    {
      dwNewV = 1;
    }
    pstOut->adwLPNumber[dwI] = dwNewV;
  }

  
  //pure experience process
  if (pstOut->dwType == LP_TYPE_DOUBLE)
  {
    int dwNowV = pstOut->adwLPNumber[pstOut->dwLPLen-1];
    //double line 挂
    if (dwNowV > 35 && dwNowV != 75)
    {
      pstOut->adwLPNumber[pstOut->dwLPLen-1] = 76;
    }
  }
  
  //pre two chinese
  dwNowV0 = pstOut->adwLPNumber[0];
  dwNowV1 = pstOut->adwLPNumber[1];
  if (dwNowV0 > 35 && dwNowV0 != 75 && dwNowV1 > 35 && dwNowV1 != 75)
  {
    for (dwI = 0; dwI < pstOut->dwLPLen-1; dwI++)
    {
      pstOut->adwLPNumber[dwI] = pstOut->adwLPNumber[dwI + 1];
      pstOut->afScores[dwI] = pstOut->afScores[dwI + 1];
    }
    
    pstOut->dwLPLen--;
  }
  
  //only WJ has 8 chars
  if (pstOut->dwLPLen==8 && (pstOut->adwLPNumber[1]!=19 && pstOut->adwLPNumber[0]!=31))
  {
    return 2;
  }
  
  //the first two chars must be alphabet or the first char must be chinese
  if (pstOut->dwLPLen>=7)
  {
    dwNowV0 = pstOut->adwLPNumber[0];
    dwNowV1 = pstOut->adwLPNumber[1];
    if (dwNowV0>=11 && dwNowV0<=35 && ((dwNowV1>=11 && dwNowV1<=35) || dwNowV1==1 || dwNowV1==2)) //military or goverment case
    {
    }
    else if (dwNowV0>=36 && ((dwNowV1>=11 && dwNowV1<=35) || dwNowV1==1 || dwNowV1==2)) //civillian case
    {
    }
    else
    {
      return 3;
    }
  }


  //there is no two province characters
  int dwNum = 0;
  for (dwI = 0; dwI < pstOut->dwLPLen; dwI++)
  {
    dwNowV0 = pstOut->adwLPNumber[dwI];
    if (dwNowV0 >= 36 && dwNowV0 <= 66)
    {
      dwNum++;
    }
  }

  if (dwNum > 1) return 4;


  //some chinese characters must be in the end of plate.
  for (dwI = 0; dwI < pstOut->dwLPLen; dwI++)
  {
    dwNowV0 = pstOut->adwLPNumber[dwI];
    if (dwNowV0 == 70 //学 
     || dwNowV0 == 73 //警
     || dwNowV0 == 74 //港
     || dwNowV0 == 76 //挂
     || dwNowV0 == 77 //澳
     )
    {
      if (dwI != pstOut->dwLPLen - 1)
      {
        return 5;
      }
    }
  }

  //mainlad only has 7 or 8 chars, but we allow it has 6 at least.
  if (pstOut->dwLPLen < 6 || pstOut->dwLPLen > 8) return 1;
  
  return 0;
}


int doRectifyWithPolyReg(LPDR_HANDLE hPolyReg, InputInfoRecog_S *pstIIR, int adwMRatioXY[2], float fAngle_old, float *pfAngle_new)
{
#if LPDR_TIME&0
  float costtime, diff;
  struct timeval start, end;

#endif
#if LPDR_TIME&0
  gettimeofday(&start, NULL);
#endif
  int dwRI;
  float *pfImage_0 = pstIIR->pfImage_0;
  float *pfImage_1 = pstIIR->pfImage_1;
  int dwImgW = pstIIR->dwW;
  int dwImgH = pstIIR->dwH;
  
  LPRect rectnew = pstIIR->rect;
  char *pbyBuffer = pstIIR->pbyBuffer;
  int dwBufferLen = pstIIR->dwBufferLen;
  float *pfCrop = 0;

  calcNewMarginBB(dwImgH, dwImgW, &rectnew, adwMRatioXY);
  
  int dwCrop_X0 = rectnew.dwX0;
  int dwCrop_Y0 = rectnew.dwY0;
  int dwCrop_X1 = rectnew.dwX1;
  int dwCrop_Y1 = rectnew.dwY1;
  int dwCrop_W = dwCrop_X1 - dwCrop_X0 + 1;
  int dwCrop_H = dwCrop_Y1 - dwCrop_Y0 + 1;
  
//  assert(dwCrop_W * dwCrop_H * 4 <= dwBufferLen);
  if (dwCrop_W * dwCrop_H * 4 > dwBufferLen) return -1;

  pfCrop = (float*)pbyBuffer;
  for (dwRI = 0; dwRI < dwCrop_H; dwRI++)
  {
    memcpy(pfCrop + dwRI * dwCrop_W, pfImage_1 + (dwRI + dwCrop_Y0) * dwImgW + dwCrop_X0, sizeof(float) * dwCrop_W);
  }

#if LPDR_TIME&0
  gettimeofday(&end, NULL);
  diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
  printf("doRectifyWithPolyReg_0 cost:%.2fms\n", diff);
#endif

#if LPDR_TIME&0
  gettimeofday(&start, NULL);
#endif
  LPDR_ImageInner_S stImage;
  stImage.pfData = pfCrop;
  stImage.dwImgW = dwCrop_W;
  stImage.dwImgH = dwCrop_H;
  int adwPolygonOut[12];
  LPPREG_Process(hPolyReg, &stImage, adwPolygonOut);
#if LPDR_TIME&0
  gettimeofday(&end, NULL);
  diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
  printf("doRectifyWithPolyReg_1 cost:%.2fms\n", diff);
#endif

#if LPDR_DBG
  {
    cv::Mat gimg(dwCrop_H, dwCrop_W, CV_32FC1, pfCrop);
    cv::Mat cimg(dwCrop_H, dwCrop_W, CV_32FC3);
    cv::cvtColor(gimg, cimg, CV_GRAY2BGR);
    cv::line(cimg, cv::Point(adwPolygonOut[0], adwPolygonOut[1]), cv::Point(adwPolygonOut[2], adwPolygonOut[3]), CV_RGB(255, 0, 0), 1, 8, 0);
    cv::imshow("doRectifyWithPolyReg_0", cimg);
    cv::waitKey(0);
  }
#endif
#if LPDR_TIME&0
  gettimeofday(&start, NULL);
#endif
  adwPolygonOut[0*2+0] += dwCrop_X0; adwPolygonOut[0*2+1] += dwCrop_Y0;
  adwPolygonOut[1*2+0] += dwCrop_X0; adwPolygonOut[1*2+1] += dwCrop_Y0;
  adwPolygonOut[2*2+0] += dwCrop_X0; adwPolygonOut[2*2+1] += dwCrop_Y0;
  adwPolygonOut[3*2+0] += dwCrop_X0; adwPolygonOut[3*2+1] += dwCrop_Y0;
  adwPolygonOut[4*2+0] += dwCrop_X0; adwPolygonOut[4*2+1] += dwCrop_Y0;
  adwPolygonOut[5*2+0] += dwCrop_X0; adwPolygonOut[5*2+1] += dwCrop_Y0;
  
//  return 0;
  doRectify_f6(pfImage_0, pfImage_1, dwImgW, dwImgH, fAngle_old, adwPolygonOut, pfAngle_new);
  
  LPRect &rectnow = pstIIR->rect;
  rectnow.dwX0 = adwPolygonOut[0]; rectnow.dwY0 = adwPolygonOut[1];
  rectnow.dwX1 = adwPolygonOut[0]; rectnow.dwY1 = adwPolygonOut[1];
  for (dwRI = 1; dwRI < 4; dwRI++)
  {
    int dwX = adwPolygonOut[dwRI*2+0];
    int dwY = adwPolygonOut[dwRI*2+1];
    if (rectnow.dwX0 > dwX) rectnow.dwX0 = dwX;
    if (rectnow.dwY0 > dwY) rectnow.dwY0 = dwY;
    if (rectnow.dwX1 < dwX) rectnow.dwX1 = dwX;
    if (rectnow.dwY1 < dwY) rectnow.dwY1 = dwY;
  }
  
  int dwSepY = min(adwPolygonOut[4*2+1], adwPolygonOut[5*2+1]);
  if (dwSepY < rectnow.dwY0) dwSepY = rectnow.dwY0;
  pstIIR->dwSepY = dwSepY;
  
#if LPDR_TIME&0
  gettimeofday(&end, NULL);
  diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
  printf("doRectifyWithPolyReg_2 cost:%.2fms\n", diff);
#endif
#if LPDR_DBG
  {
    cv::Mat gimg(dwImgH, dwImgW, CV_32FC1, pfImage_1);
    cv::Mat cimg(dwImgH, dwImgW, CV_32FC3);
    cv::cvtColor(gimg, cimg, CV_GRAY2BGR);
    cv::rectangle(cimg, cv::Point(rectnow.dwX0, rectnow.dwY0), cv::Point(rectnow.dwX1, rectnow.dwY1), CV_RGB(255, 0, 0), 1, 8, 0);
    cv::imshow("doRectifyWithPolyReg_1", cimg);
    cv::waitKey(0);
  }
#endif
  return 0;
}


int doRecogOneRow(LPDR_HANDLE hChRecog, LPDR_ImageInner_S *pstImage, LPRect rect, float fStrechRatio, float fShrinkRatio, int dwStep, float fThreshold, LPDRInfo_S *pstOut)
{
#if LPDR_TIME&0
  float costtime, diff;
  struct timeval start, end;

#endif
#if LPDR_TIME&0
  gettimeofday(&start, NULL);
#endif
  int dwBestRet = 0;
  int dwI, dwJ, dwII0, dwII1;
  ModuleCHRECOG_S *pstCHRECOG = (ModuleCHRECOG_S*)hChRecog;
  int adwStdHW[2] = {pstCHRECOG->adwInShape[2], pstCHRECOG->adwInShape[3]};
  
  memset(pstOut, 0, sizeof(LPDRInfo_S));
  int dwRectH = rect.dwY1 - rect.dwY0 + 1;
  
  //multiple times tests
#if 0
  const int dwUpNum = 2, dwDnNum = 2;
  int adwMarginUps[dwUpNum] = {0, dwRectH/16};
  int adwMarginDns[dwDnNum] = {dwRectH/16, dwRectH/8};
#else
  const int dwUpNum = 1, dwDnNum = 1;
  int adwMarginUps[dwUpNum] = {0};
  int adwMarginDns[dwDnNum] = {dwRectH/8};
#endif
  for (dwII0 = 0; dwII0 < dwUpNum; dwII0++)
  {
    int dwMgnUp = adwMarginUps[dwII0];
    for (dwII1 = 0; dwII1 < dwDnNum; dwII1++)
    {
      int dwMgnDn = adwMarginDns[dwII1];
      LPDRInfo_S stLPOut;

      memset(&stLPOut, 0, sizeof(LPDRInfo_S));

      int adwMarginLRTD[4] = {adwStdHW[1]/16, adwStdHW[1]/16, dwMgnUp, dwMgnDn}; //left, right, top, down
//      int adwMarginLRTD[4] = {adwStdHW[1], adwStdHW[1], dwMgnUp, dwMgnDn}; //left, right, top, down
      LPRect newRect;
      newRect.dwX0 = max(rect.dwX0 - adwMarginLRTD[0], 0);
      newRect.dwY0 = max(rect.dwY0 - adwMarginLRTD[2], 0);
      newRect.dwX1 = min(rect.dwX1 + adwMarginLRTD[1], pstImage->dwImgW - 1);
      newRect.dwY1 = min(rect.dwY1 + adwMarginLRTD[3], pstImage->dwImgH - 1);

      LPCHRECOG_Process(hChRecog, pstImage, newRect, fStrechRatio, fShrinkRatio, dwStep);

      float *pfOutScore = pstCHRECOG->pfOutScore;
      int *pdwOutShape = pstCHRECOG->adwOutShape;
      int dwBatchNum = pdwOutShape[0];
      int dwClassNum = pdwOutShape[1];
      int *pdwClassIdx = pstCHRECOG->pdwClassIdx;
      float *pfClassScore = pstCHRECOG->pfClassScore;
      memset(pdwClassIdx, 0, sizeof(int)*dwBatchNum);
      memset(pfClassScore, 0, sizeof(float)*dwBatchNum);
      for (dwI = 0; dwI < dwBatchNum; dwI++)
      {
        float *pfScoreRow = pfOutScore + dwI * dwClassNum;
        float fMaxScore = pfScoreRow[0];
        int dwMaxIdx = 0;
        for (dwJ = 1; dwJ < dwClassNum; dwJ++)
        {
          if (fMaxScore < pfScoreRow[dwJ])
          {
            fMaxScore = pfScoreRow[dwJ];
            dwMaxIdx = dwJ;
          }
        }
        pdwClassIdx[dwI] = dwMaxIdx;
        pfClassScore[dwI] = fMaxScore;
      }

      int dwRet = parseRecogOutInfo(pdwClassIdx, pfClassScore, dwBatchNum, fThreshold, &stLPOut);

    #if LPDR_DBG
      cout << stLPOut.fAllScore << ": ";
      for (dwI = 0; dwI < dwBatchNum; dwI++)
      {
        cout << paInv_chardict[pdwClassIdx[dwI]];
      }
      
      cout << endl;
    #endif
      
      if (stLPOut.fAllScore > pstOut->fAllScore)
      {
        *pstOut = stLPOut;
        dwBestRet = dwRet;
      }
    }
  }
#if LPDR_TIME&0
  gettimeofday(&end, NULL);
  diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
  printf("doRecogOneRow cost:%.2fms\n", diff);
#endif
//  cout << pstOut->dwLPLen << endl;
  return dwBestRet;
}


int parseRecogOutInfo(int *pdwClassIdx, float *pfClassScore, int dwNum, float fThreshold, LPDRInfo_S *pstOut)
{
  int dwChNum;
  int dwPI, dwJ;
  int dwStart, dwEnd;
  float afClassScore[LPDR_CLASS_NUM];
  int adwClassNum[LPDR_CLASS_NUM];
  int dwClsIdx;
  int dwMaxIdx;
  float fMaxScore;
  int *pdwLPNumber = pstOut->adwLPNumber;
  float *pfScores = pstOut->afScores, fAllScore;
  int dwLen = 0;
  
  fAllScore = 0.f;
  dwStart = -1;
  dwEnd = -1;
  for (dwPI = 0; dwPI < dwNum; dwPI++)
  {
    if (pdwClassIdx[dwPI] != 0 && dwPI < dwNum-1)
    {
      if (dwStart == -1) dwStart = dwPI;
    }
    else if (dwStart != -1)
    {
      dwEnd = dwPI;
      memset(afClassScore, 0, sizeof(float) * LPDR_CLASS_NUM);
      memset(adwClassNum, 0, sizeof(int) * LPDR_CLASS_NUM);
      for (dwJ = dwStart; dwJ < dwEnd; dwJ++)
      {
        dwClsIdx = pdwClassIdx[dwJ];
        adwClassNum[dwClsIdx]++;
        afClassScore[dwClsIdx] += pfClassScore[dwJ];
      }
      
      fMaxScore = afClassScore[0];
      dwMaxIdx = 0;
      for (dwJ = 1; dwJ < LPDR_CLASS_NUM; dwJ++)
      {
        if (fMaxScore < afClassScore[dwJ])
        {
          fMaxScore = afClassScore[dwJ];
          dwMaxIdx = dwJ;
        }
      }
      pdwLPNumber[dwLen] = dwMaxIdx;
      pfScores[dwLen] = fMaxScore/adwClassNum[dwMaxIdx];
      fAllScore += fMaxScore;
      dwLen++;
      if (dwLen >= MAX_LPCHAR_NUM)
      {
        break;
      }
//      cout << dwLen << ", " << dwMaxIdx <<  endl;
      dwStart = -1;
      dwEnd = -1;
    }
  }
//  cout << dwLen << endl;
  
  if (fAllScore < fThreshold)
  {
#if LPDR_DBG
    cout << fAllScore << ", " << fThreshold << endl;
#endif
//    cout << "fuck you!\n";
    pstOut->dwLPLen = 0;
    return 1;
  }
  
//  cout << dwLen << endl;
  pstOut->dwLPLen = dwLen;
  pstOut->fAllScore = fAllScore;
  
  return 0;
}


#if 0
/*
#define LP_COLOUR_UNKNOWN   0
#define LP_COLOUR_BLUE      1
#define LP_COLOUR_YELLOW    2
#define LP_COLOUR_WHITE     3
#define LP_COLOUR_BLACK     4
#define LP_COLOUR_GREEN     5
*/
int doRecogColors(LPDR_HANDLE handle, LPDR_ImageSet_S *pstImgSet, LPDR_OutputSet_S *pstOutputSet)
{
  int dwSI, dwLI, dwRI, dwCI;
  int dwImgNum = pstImgSet->dwImageNum;
  int dwImgW, dwImgH;
  uchar *pubyImgData, *pubyRow, *pubyBGR;
  int dwLPNum = 0;
  LPDRInfo_S *pstLPDR = 0;
  int adwBB[4];
  float fH = 0.f, fS = 0.f, fV = 0.f;
  int adwColorsHist[9], dwMaxColor, dwMaxValue;
  
  for (dwSI = 0; dwSI < dwImgNum; dwSI++)
  {
    dwImgW = pstImgSet->astSet[dwSI].dwImgW;
    dwImgH = pstImgSet->astSet[dwSI].dwImgH;
    pubyImgData = pstImgSet->astSet[dwSI].pubyData;
    dwLPNum = pstOutputSet->astLPSet[dwSI].dwLPNum;
    for (dwLI = 0; dwLI < dwLPNum; dwLI++)
    {
      pstLPDR = &pstOutputSet->astLPSet[dwSI].astLPs[dwLI];
      adwBB[0] = pstLPDR->adwLPRect[0];
      adwBB[1] = pstLPDR->adwLPRect[1];
      adwBB[2] = pstLPDR->adwLPRect[2];
      adwBB[3] = pstLPDR->adwLPRect[3];
      memset(adwColorsHist, 0, sizeof(int)*9);
      for (dwRI = adwBB[1]; dwRI < adwBB[3]; dwRI += 2)
      {
        pubyRow = pubyImgData + dwRI * 3 * dwImgW;
        for (dwCI = adwBB[0]; dwCI < adwBB[2]; dwCI += 2)
        {
          pubyBGR = pubyRow + dwCI * 3;
          cvtRGB2HSV_U8(pubyBGR[2], pubyBGR[1], pubyBGR[0], &fH, &fS, &fV);
          if (fS > 0.16f && fV > 0.08)
          {
//            if (fH > 30.f && fH < 50.f) //0~240
            if (fH > 45.f && fH < 75.f) //0~360
            {
              adwColorsHist[LP_COLOUR_YELLOW]++;
            }
//            else if (fH > 130.f && fH < 180.f)
            else if (fH > 195.f && fH < 270.f)
            {
              adwColorsHist[LP_COLOUR_BLUE]++;
            }
//            else if (fH > 60.f && fH < 100.f)
            else if (fH > 90.f && fH < 150.f)
            {
              adwColorsHist[LP_COLOUR_GREEN]++;
            }
          }
          else if (fS < 0.16f && fV > 0.4)
          {
            adwColorsHist[LP_COLOUR_WHITE]++;
          }
          else if (fS < 0.16f && fV < 0.4)
          {
            adwColorsHist[LP_COLOUR_BLACK]++;
          }
        }
      }
#if LPDR_DBG
      for (int dwI = 0; dwI < 9; dwI++)
      {
        printf("%d:%d, ", dwI, adwColorsHist[dwI]);
      }
      printf("\n");
#endif
      dwMaxColor = 0;
      dwMaxValue = 0;
      for (int dwI = 0; dwI < 9; dwI++)
      {
        if (dwMaxValue < adwColorsHist[dwI])
        {
          dwMaxValue = adwColorsHist[dwI];
          dwMaxColor = dwI;
        }
      }
      pstLPDR->dwColor = dwMaxColor;
    }

#if LPDR_DBG||1
//    cv::Mat inputColorOne(dwImgH, dwImgW, CV_8UC3, );
    
#endif
  }
  
  return 0;
}
#endif


/*
#define LP_COLOUR_UNKNOWN   0
#define LP_COLOUR_BLUE      1
#define LP_COLOUR_YELLOW    2
#define LP_COLOUR_WHITE     3
#define LP_COLOUR_BLACK     4
#define LP_COLOUR_GREEN     5
*/
int doRecogColors(LPDR_HANDLE handle, LPDR_ImageSet_S *pstImgSet, LPDR_OutputSet_S *pstOutputSet)
{
  int dwSI, dwLI, dwRI, dwCI;
  int dwImgNum = pstImgSet->dwImageNum;
  int dwImgW, dwImgH;
  uchar *pubyImgData, *pubyRow, *pubyBGR;
  int dwLPNum = 0;
  LPDRInfo_S *pstLPDR = 0;
  int adwBB[4];
  float fH = 0.f, fS = 0.f, fV = 0.f;
  int adwColorsHist[6], adwIdxs[6], dwTmpValue;
  
  for (dwSI = 0; dwSI < dwImgNum; dwSI++)
  {
    dwImgW = pstImgSet->astSet[dwSI].dwImgW;
    dwImgH = pstImgSet->astSet[dwSI].dwImgH;
    pubyImgData = pstImgSet->astSet[dwSI].pubyData;
    dwLPNum = pstOutputSet->astLPSet[dwSI].dwLPNum;
    for (dwLI = 0; dwLI < dwLPNum; dwLI++)
    {
      pstLPDR = &pstOutputSet->astLPSet[dwSI].astLPs[dwLI];
      int dwBBH = pstLPDR->adwLPRect[3] - pstLPDR->adwLPRect[1] + 1;
      int dwBBW = pstLPDR->adwLPRect[2] - pstLPDR->adwLPRect[0] + 1;
      adwBB[0] = pstLPDR->adwLPRect[0] + dwBBW/12;
      adwBB[1] = pstLPDR->adwLPRect[1] + dwBBH/8;
      adwBB[2] = pstLPDR->adwLPRect[2] - dwBBW/12;
      adwBB[3] = pstLPDR->adwLPRect[3] - dwBBH/8;
      memset(adwColorsHist, 0, sizeof(int)*6);
      for (dwRI = adwBB[1]; dwRI < adwBB[3]; dwRI += 2)
      {
        pubyRow = pubyImgData + dwRI * 3 * dwImgW;
        for (dwCI = adwBB[0]; dwCI < adwBB[2]; dwCI += 2)
        {
          pubyBGR = pubyRow + dwCI * 3;
          cvtRGB2HSV_U8(pubyBGR[2], pubyBGR[1], pubyBGR[0], &fH, &fS, &fV);
//          printf("%.2f,%.2f,%.2f; ", fH, fS, fV);
//          if (fS > 0.16f && fV > 0.10)
          if (fS > 0.10f && fV > 0.10)
          {
//            if (fH > 30.f && fH < 50.f) //0~240
            if (fH > 20.f && fH < 75.f) //0~360
            {
              adwColorsHist[LP_COLOUR_YELLOW]++;
            }
//            else if (fH > 130.f && fH < 180.f)
            else if (fH > 100.f && fH < 290.f)
            {
              adwColorsHist[LP_COLOUR_BLUE]++;
            }
//            else if (fH > 60.f && fH < 100.f)
            else if (fH > 90.f && fH < 150.f)
            {
              adwColorsHist[LP_COLOUR_GREEN]++;
            }
          }
          else if (fS < 0.45f && fV > 0.4)
//          else if (fV > 0.3)
          {
            adwColorsHist[LP_COLOUR_WHITE]++;
          }
          else if (fS < 0.45f && fV < 0.4)
//          else if (fV < 0.3)
          {
            adwColorsHist[LP_COLOUR_BLACK]++;
          }
        }
      }
      
      for (int dwI = 0; dwI < 6; dwI++)
      {
        adwIdxs[dwI] = dwI;
      }
      
      dwTmpValue = 0;
      for (int dwI = 0; dwI < 5; dwI++)
      {
        for (int dwJ = dwI + 1; dwJ < 6; dwJ++)
        {
          if (adwColorsHist[dwI] < adwColorsHist[dwJ])
          {
            dwTmpValue = adwColorsHist[dwI];
            adwColorsHist[dwI] = adwColorsHist[dwJ];
            adwColorsHist[dwJ] = dwTmpValue;
            
            dwTmpValue = adwIdxs[dwI];
            adwIdxs[dwI] = adwIdxs[dwJ];
            adwIdxs[dwJ] = dwTmpValue;
          }
        }
      }
      
#if LPDR_DBG
      cv::Mat cimg(dwImgH, dwImgW, CV_8UC3, pubyImgData);
      cv::Mat subcimg = cimg(cv::Range(adwBB[1], adwBB[3]), cv::Range(adwBB[0], adwBB[2]));
      cv::imshow("hello", subcimg);
      cv::waitKey(10);
      string astrColors[6] = {"0.UNKNOWN", "1.BLUE", "2.YELLOW", "3.WHITE", "4.BLACK", "5.GREEN"};
      for (int dwI = 0; dwI < 6; dwI++)
      {
        printf("%s:%d, ", astrColors[adwIdxs[dwI]].c_str(), adwColorsHist[dwI]);
      }
      printf("\n");
#endif
      int dwMaxColor = adwIdxs[0];
      
      if ((adwIdxs[0] == LP_COLOUR_WHITE || adwIdxs[0] == LP_COLOUR_BLACK || adwIdxs[0] == LP_COLOUR_GREEN) && (adwIdxs[1] == LP_COLOUR_YELLOW || adwIdxs[1] == LP_COLOUR_BLUE || adwIdxs[1] == LP_COLOUR_GREEN) && adwColorsHist[0]*40 < adwColorsHist[1]*100)
      {
        dwMaxColor = adwIdxs[1];
      }

      for (int k = 0; k < 6; k++)
      {
        if (adwIdxs[k] == LP_COLOUR_BLUE && adwColorsHist[k] * 80 > adwColorsHist[0])
        {
          dwMaxColor = LP_COLOUR_BLUE;
        }
      }
      
      pstLPDR->dwColor = dwMaxColor;
    }

#if LPDR_DBG||1
//    cv::Mat inputColorOne(dwImgH, dwImgW, CV_8UC3, );
    
#endif
  }
  
  return 0;
}









