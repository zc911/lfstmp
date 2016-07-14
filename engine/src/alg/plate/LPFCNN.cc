
#include "LPFCNN.hpp"


int getRectsOfFCNN(float *pfScore, int dwImgH, int dwImgW, vector<LPRectInfo> &lprects);


int LPFCNN_Create(LPDRModel_S stFCNN, int dwDevType, int dwDevID, LPDR_HANDLE *phFCNN)
{
  ModuleFCNN_S *pstModule = (ModuleFCNN_S*)calloc(1, sizeof(ModuleFCNN_S));
  *phFCNN = (LPDR_HANDLE)pstModule;
  
  SymbolHandle hSymbol = 0;
  int ret = 0;

  //load model
//  cout << "fcnn 0\n";
//  cout << strlen(stFCNN.pbySym) << endl;
  ret = MXSymbolCreateFromJSON(stFCNN.pbySym, &hSymbol);
//  cout << "fcnn 1\n";
  pstModule->hSymbol = hSymbol;
  assert(ret==0);
#if DR_DBG&0
  cout << ret << endl;
#endif

  //infer_shape
  mx_uint num_args = 1;
  const mx_uint *pdwShape = (mx_uint*)stFCNN.adwShape;
  const char *keys[] = {"data"};
  const mx_uint arg_ind_ptr[] = {0, 4};
  const mx_uint arg_shape_data[] = {pdwShape[0], pdwShape[1], pdwShape[2], pdwShape[3]};
  mx_uint in_shape_size = 0;
  const mx_uint *in_shape_ndim = 0;
  const mx_uint **in_shape_data = 0;
  mx_uint out_shape_size = 0;
  const mx_uint *out_shape_ndim = 0;
  const mx_uint **out_shape_data = 0;
  mx_uint aux_shape_size = 0;
  const mx_uint *aux_shape_ndim = 0;
  const mx_uint **aux_shape_data = 0;
  int complete = 0;

  ret = MXSymbolInferShape(hSymbol, num_args, keys, arg_ind_ptr, arg_shape_data,
              &in_shape_size, &in_shape_ndim, &in_shape_data,
              &out_shape_size, &out_shape_ndim, &out_shape_data,
              &aux_shape_size, &aux_shape_ndim, &aux_shape_data, &complete);

#if DR_DBG&0
   cout << ret << endl;
   cout << "==in_shape info:" << endl;
   cout << in_shape_size << endl;
   for (int i = 0; i < in_shape_size; i++) {
    cout << in_shape_ndim[i] << "-->";
    for (int j = 0; j < in_shape_ndim[i]; j++) {
     cout << in_shape_data[i][j] << ", ";
    }
    cout << endl;
   }
   
   cout << "==out_shape info:" << endl;
   cout << out_shape_size << endl;
   for (int i = 0; i < out_shape_size; i++) {
    cout << out_shape_ndim[i] << "-->";
    for (int j = 0; j < out_shape_ndim[i]; j++) {
     cout << out_shape_data[i][j] << ", ";
    }
    cout << endl;
   }
#endif
#if DR_DBG&0
 cout << "aux_shape info:" << endl;
 cout << aux_shape_size << endl;
 for (int i = 0; i < aux_shape_size; i++) {
  cout << aux_shape_ndim[i] << "-->";
  for (int j = 0; j < aux_shape_ndim[i]; j++) {
   cout << aux_shape_data[i][j] << ", ";
  }
  cout << endl;
 }
#endif

  //load parameters from trained file.
  unordered_map<string, NDArrayHandle> param_pairs;
  mx_uint param_size = 0;
  NDArrayHandle* paramh_arr = 0;
  mx_uint param_name_size = 0;
  const char** param_names = 0;
  ret = MXNDArrayLoadFromBytes(stFCNN.pbyParam, stFCNN.dwParamSize, &param_size, &paramh_arr, &param_name_size, &param_names);
  assert(param_size == param_name_size);
  for (int i = 0 ; i < param_name_size; i++) {
#if DR_DBG&0
    cout << i << ":" << param_names[i] << endl;
#endif
    param_pairs[param_names[i]] = paramh_arr[i];
  }

  //list arguments
  mx_uint args_size = 0;
  const char **args_str_array = 0;
  ret = MXSymbolListArguments(hSymbol, &args_size, &args_str_array);
#if DR_DBG&0
  cout << ret << endl;
  cout << args_size << endl;
  for (int i = 0; i < args_size; i++) {
   cout << i << ":" << args_str_array[i] << endl;
  }
#endif

  //create parameter ndarray
 mx_uint len = args_size;

 NDArrayHandle *in_args = (NDArrayHandle*)calloc(len, sizeof(NDArrayHandle));
 NDArrayHandle *arg_grad_store = (NDArrayHandle*)calloc(len, sizeof(NDArrayHandle));
 mx_uint *grad_req_type = (mx_uint*)calloc(len, sizeof(mx_uint));
 mx_uint aux_states_len = 0;
 NDArrayHandle *aux_states = 0;//(NDArrayHandle*)calloc(len, sizeof(NDArrayHandle));
// cout << in_shape_size << "," << len << endl;
  assert(in_shape_size==len);

  for (int i = 0; i < in_shape_size; i++) {
    ret = MXNDArrayCreate(in_shape_data[i], in_shape_ndim[i], dwDevType, dwDevID, 0, &in_args[i]);
#if DR_DBG&0
    cout << i << ":" << ret << ", ";
#endif
  }
#if DR_DBG&0
  cout << endl;
#endif

  //copy trained parameters into created parameter ndarray.
  const size_t maxbuffer_size = 128 * 128 * 128;
  mx_float *pfBuffer = (mx_float*)calloc(maxbuffer_size, sizeof(mx_float));
  for (int i = 0; i < args_size; i++) {
    string name = args_str_array[i];
//   cout << i << ":" << name << "--" << param_pairs[name] << endl;
   if (param_pairs[name]) {
     NDArrayHandle hparam = param_pairs[name];
     NDArrayHandle hparamto = in_args[i];
     copy_ndarray(hparam, hparamto, pfBuffer, maxbuffer_size);
   }
  }
  free(pfBuffer);
  //free loaded parameters ndarray
  //cout << args_size << ", " << param_size << endl;
  for (int i = 0; i < param_size; i++) {
    MXNDArrayFree(paramh_arr[i]);
  }
#if DR_DBG&0
  for (int i = 0; i < args_size; i++) {
    cout << i << ":" << in_args[i] << endl;
  }
  cout << ret << endl;
#endif

 //do bind
  ret = MXExecutorBind(hSymbol, dwDevType, dwDevID,
             len, in_args, arg_grad_store,
             grad_req_type, aux_states_len,
             aux_states, &pstModule->hExecute);

  pstModule->args_arr = in_args;
  pstModule->args_num = args_size;

  ret = MXExecutorForward(pstModule->hExecute, 0);
#if DR_DBG&0
 cout << ret << endl;
#endif

  /////////////////////////////////////////////////////////////
  
  for (int i = 0; i < 4; i++)
  {
    pstModule->adwInShape[i] = stFCNN.adwShape[i];
  }
  
  int *pdwInShape = pstModule->adwInShape;
  int dwInputSize = pdwInShape[0] * pdwInShape[1] * pdwInShape[2] * pdwInShape[3];
  
  pstModule->pubyInputData = (uchar*)calloc(dwInputSize, 1);
  pstModule->pfInputData = (float*)calloc(dwInputSize, sizeof(float));

  pstModule->dwBuffSize = 1024*1024*1024;
  pstModule->pubyBuffer = (uchar*)calloc(pstModule->dwBuffSize, 1);
  
  mx_uint out_size = 0;
  NDArrayHandle *out = 0;
  ret = MXExecutorOutputs(pstModule->hExecute, &out_size, &out);
  
  int *pdwOutShape = pstModule->adwOutShape;
  
  mx_uint out_dim = 0;
  const mx_uint *out_shape = 0;
  NDArrayHandle hout = out[0];
  ret = MXNDArrayGetShape(hout, &out_dim, &out_shape);
  size_t needsize = 1;
  for (int j = 0; j < out_dim; j++) {
    needsize *= out_shape[j];
    pdwOutShape[j] = out_shape[j];
  }
  
  for (int j = 0; j < out_size; j++)
  {
    MXNDArrayFree(out[j]);
  }
  
  int dwOutputSize = pdwOutShape[0] * pdwOutShape[1] * pdwOutShape[2] * pdwOutShape[3];
  
  pstModule->pfOutputData = (float*)calloc(dwOutputSize, sizeof(float));
  
  pstModule->dwCheckW = 300;
  pstModule->dwCheckH = 100;
  
  pstModule->pdwRealWs = new int[pdwInShape[0]];
  memset(pstModule->pdwRealWs, 0, sizeof(int) * pdwInShape[0]);
  pstModule->pdwRealHs = new int[pdwInShape[0]];
  memset(pstModule->pdwRealHs, 0, sizeof(int) * pdwInShape[0]);
  
  pstModule->plpgroup = new vector<LPRectInfo>[pdwInShape[0]];
  
  return 0;
}


int LPFCNN_Process(LPDR_HANDLE hFCNN, LPDR_ImageInner_S *pstImgSet, int dwImgNum)
{
  int dwI, dwJ;
  int ret;
#if LPDR_TIME
  float costtime, diff;
  struct timeval start, end;

#endif

  ModuleFCNN_S *pstFCNN = (ModuleFCNN_S*)hFCNN;
  
  int dwCheckW = pstFCNN->dwCheckW;
  int dwCheckH = pstFCNN->dwCheckH;

  NDArrayHandle hData = pstFCNN->args_arr[0];
  
  int *pdwInShape = pstFCNN->adwInShape;
  int dwStdH = pdwInShape[2];
  int dwStdW = pdwInShape[3];
  int dwRealW, dwRealH;
  int needsize = getSize(hData);
  int dwImgWOri, dwImgHOri;

#if LPDR_TIME
  gettimeofday(&start, NULL);
#endif
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    vector<LPRectInfo> &lpgroup = pstFCNN->plpgroup[dwI];
    lpgroup.clear();
  
    LPDR_ImageInner_S *pstImage = &pstImgSet[dwI];
    
    float *pfDataOri = pstImage->pfData;
    
    dwImgWOri = pstImage->dwImgW;
    dwImgHOri = pstImage->dwImgH;

//    uchar *pubyStdInputData = pstFCNN->pubyInputData;
    float *pfStdInputData = pstFCNN->pfInputData + dwI * dwStdW * dwStdH;
    memset(pfStdInputData, 0, sizeof(float) * dwStdW * dwStdH);
    imgResizeAddBlack_f(pfDataOri, dwImgWOri, dwImgHOri, pfStdInputData, dwStdW, dwStdH, &dwRealW, &dwRealH);

    pstFCNN->pdwRealWs[dwI] = dwRealW;
    pstFCNN->pdwRealHs[dwI] = dwRealH;
  }
#if LPDR_TIME
	gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
	printf("in FCNN pre cost[%dx%d]:%.2fms\n", dwStdH, dwStdW, diff);
#endif

#if LPDR_TIME
  gettimeofday(&start, NULL);
#endif

  ret = MXNDArraySyncCopyFromCPU(hData, pstFCNN->pfInputData, needsize);

  ret = MXExecutorForward(pstFCNN->hExecute, 0);

  mx_uint out_size = 0;
  NDArrayHandle *out = 0;

  ret = MXExecutorOutputs(pstFCNN->hExecute, &out_size, &out);

  int *pdwOutShape = pstFCNN->adwOutShape;
  float *pfOutput = pstFCNN->pfOutputData;
  int dwNeedSize = pdwOutShape[0] * pdwOutShape[1] * pdwOutShape[2] * pdwOutShape[3];
  NDArrayHandle hout = out[0];
  
  ret = MXNDArraySyncCopyToCPU(hout, pfOutput, dwNeedSize);
	
	MXNDArrayFree(hout);
	
#if LPDR_TIME
	gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
	printf("FCNN cost[%dx%d]:%.2fms\n", dwStdH, dwStdW, diff);
#endif

	int dwOutH = pdwOutShape[2];
	int dwOutW = pdwOutShape[3];
	
#if LPDR_DBG&1
  
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    char abyChars[128];
    cv::Mat fcnnOut(dwOutH, dwOutW, CV_32FC1, pfOutput + dwI * dwOutH * dwOutW);
    sprintf(abyChars, "fcnnOut_%d", dwI);
    cv::namedWindow(abyChars, 0);
    cv::imshow(abyChars, fcnnOut);
    cv::waitKey(0);
  }
#endif
  
  
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
//    cout << "fcnn_img:" << dwI << endl;
    vector<LPRectInfo> lprects;
    getRectsOfFCNN(pfOutput + dwI * dwOutH * dwOutW, dwOutH, dwOutW, lprects);
    
    dwRealW = pstFCNN->pdwRealWs[dwI];
    dwRealH = pstFCNN->pdwRealHs[dwI];
    
    LPDR_ImageInner_S *pstImage = &pstImgSet[dwI];
    
    dwImgWOri = pstImage->dwImgW;
    dwImgHOri = pstImage->dwImgH;
    
    for (dwJ = 0; dwJ < lprects.size(); dwJ++)
    {
      LPRectInfo *prect = &lprects[dwJ];
      prect->fWidth *= 2;//3;
      prect->fHeight *= 2;//2;
    }

    vector<LPRectInfo> &lpgroup = pstFCNN->plpgroup[dwI];
//    cout << "lpgroup size:" << lpgroup.size() << endl;
//    group_bbs(lprects, lpgroup, 0.5f);
    group_bbs_overlap(lprects, lpgroup, 0.8f);
    
    float afBB[4];
    for (dwJ = 0; dwJ < lpgroup.size(); dwJ++)
    {
      LPRectInfo *prect = &lpgroup[dwJ];

      prect->fCentX = prect->fCentX * dwStdW / dwOutW * dwImgWOri / dwRealW;
      prect->fCentY = prect->fCentY * dwStdH / dwOutH * dwImgHOri / dwRealH;
      prect->fWidth = prect->fWidth * dwStdW / dwOutW * dwImgWOri / dwRealW;
      prect->fHeight = prect->fHeight * dwStdH / dwOutH * dwImgHOri / dwRealH;
      
      afBB[0] = prect->fCentX - prect->fWidth/2;
      afBB[1] = prect->fCentY - prect->fHeight/2;
      afBB[2] = prect->fCentX + prect->fWidth/2;
      afBB[3] = prect->fCentY + prect->fHeight/2;
      
      afBB[0] = afBB[0] < 0 ? 0 : afBB[0];
      afBB[1] = afBB[1] < 0 ? 0 : afBB[1];
      afBB[2] = afBB[2] >= dwImgWOri ? dwImgWOri-1 : afBB[2];
      afBB[3] = afBB[3] >= dwImgHOri ? dwImgHOri-1 : afBB[3];
      
      prect->fWidth = afBB[2] - afBB[0];
      prect->fHeight = afBB[3] - afBB[1];
      prect->fCentX = afBB[0] + prect->fWidth/2;
      prect->fCentY = afBB[1] + prect->fHeight/2;

  //    printf("%d:%.2f, %d, %d, %d, %d\n", dwI, prect->fScore, (int)prect->fCentX, (int)prect->fCentY, (int)prect->fWidth, (int)prect->fHeight);
    }
  }

#if LPDR_DBG&1
  
  for (dwI = 0; dwI < dwImgNum; dwI++)
  {
    char abyChars[128];
    LPDR_ImageInner_S *pstImage = &pstImgSet[dwI];
    float *pfDataOri = pstImage->pfData;
    
    dwImgWOri = pstImage->dwImgW;
    dwImgHOri = pstImage->dwImgH;
    
    cv::Mat gimg(dwImgHOri, dwImgWOri, CV_32FC1, pfDataOri);
    cv::Mat cimg;
    cv::cvtColor(gimg, cimg, CV_GRAY2BGR);
    vector<LPRectInfo> &lpgroup = pstFCNN->plpgroup[dwI];
    
    for (dwJ = 0; dwJ < lpgroup.size(); dwJ++)
    {
      LPRectInfo *prect = &lpgroup[dwJ];

      int dwX0 = prect->fCentX - prect->fWidth / 2;
      int dwY0 = prect->fCentY - prect->fHeight / 2;
      int dwX1 = prect->fCentX + prect->fWidth / 2;
      int dwY1 = prect->fCentY + prect->fHeight / 2;
      
      cv::rectangle(cimg, cv::Point(dwX0, dwY0), cv::Point(dwX1, dwY1), CV_RGB(255, 0, 0), 2, 8, 0);
    }
    sprintf(abyChars, "fcnnOut_%d", dwI);
    cv::namedWindow(abyChars, 0);
    cv::imshow(abyChars, cimg);
    cv::waitKey(10);
  }
#endif

  return 0;
}


int LPFCNN_Release(LPDR_HANDLE hFCNN)
{
  int ret = 0;
  
  ModuleFCNN_S *pstFCNN = (ModuleFCNN_S*)hFCNN;
  
  ret = MXSymbolFree(pstFCNN->hSymbol);
	ret = MXExecutorFree(pstFCNN->hExecute);
	
	for (int i = 0; i < pstFCNN->args_num; i++) {
    ret = MXNDArrayFree(pstFCNN->args_arr[i]);
  }
//  cout << "release fcnn 0\n";
  free(pstFCNN->pubyInputData);
  
//  cout << "release fcnn 1\n";
  free(pstFCNN->pfInputData);
  
//  cout << "release fcnn 2\n";
  free(pstFCNN->pubyBuffer);
  
//  cout << "release fcnn 3\n";
  free(pstFCNN->pfOutputData);
  
  delete []pstFCNN->pdwRealWs;
  delete []pstFCNN->pdwRealHs;
  delete []pstFCNN->plpgroup;
  
  free(pstFCNN);
  
  return 0;
}


//int group_bbs(vector<LPRectInfo> &lprects, vector<LPRectInfo> &group)

int getRectsOfFCNN(float *pfScore, int dwImgH, int dwImgW, vector<LPRectInfo> &lprects)
{
  int dwI, dwRI, dwCI;
  int dwSize = dwImgH * dwImgW;
  
  uchar *pubyBinImg = (uchar*)calloc(dwImgH * dwImgW, 1);
  uchar *pubyBinImg2 = (uchar*)calloc(dwImgH * dwImgW, 1);
  
  for (dwI = 0; dwI < dwSize; dwI++)
  {
    if (pfScore[dwI] > 0.8) pubyBinImg[dwI] = 255;
  }
  memcpy(pubyBinImg2, pubyBinImg, dwSize);
  
  cv::Mat binImg(dwImgH, dwImgW, CV_8UC1, pubyBinImg);
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;

#if LPDR_DBG
  cv::imshow("binImg", binImg);
  cout<<"hi binImg\n";
  cv::waitKey(0);
#endif

  cv::findContours(binImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

  for(dwI = 0 ; dwI < contours.size(); dwI++)
  {
    cv::Rect rect = cv::boundingRect(contours[dwI]);
//    cout << rect << endl;
    int x = rect.x;
    int y = rect.y;
    int w = rect.width;
    int h = rect.height;
    if (w < 2 || h < 2) continue;
    
    float fScore = 0.0;
    int dwNum = 1;
    for (dwRI = y; dwRI < y + h; dwRI++)
    {
      float *pfRow = pfScore + dwRI * dwImgW;
      uchar *pubyRow = pubyBinImg2 + dwRI * dwImgW;
      for (dwCI = x; dwCI < x + w; dwCI++)
      {
        if (pubyRow[dwCI] > 0)
        {
          fScore += pfRow[dwCI];
          dwNum++;
        }
      }
    }
    fScore /= dwNum;
    LPRectInfo lprect(fScore, y+h/2, x+w/2, h, w);
    lprects.push_back(lprect);
  }
  
  free(pubyBinImg);
  free(pubyBinImg2);
  
  return 0;
}







