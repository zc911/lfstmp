
#include "LPCHRECOG.hpp"


int LPCHRECOG_Create(LPDRModel_S stCHRECOG, int dwDevType, int dwDevID, LPDR_HANDLE *phCHRECOG)
{
  ModuleCHRECOG_S *pstModule = (ModuleCHRECOG_S*)calloc(1, sizeof(ModuleCHRECOG_S));
  *phCHRECOG = (LPDR_HANDLE)pstModule;
  
  SymbolHandle hSymbol = 0;
  int ret = 0;

  //load model
  ret = MXSymbolCreateFromJSON(stCHRECOG.pbySym, &hSymbol);
  
  pstModule->hSymbol = hSymbol;
  assert(ret==0);
#if DR_DBG&0
  cout << ret << endl;
#endif

  //infer_shape
  mx_uint num_args = 1;
  const mx_uint *pdwShape = (mx_uint*)stCHRECOG.adwShape;
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
  ret = MXNDArrayLoadFromBytes(stCHRECOG.pbyParam, stCHRECOG.dwParamSize, &param_size, &paramh_arr, &param_name_size, &param_names);
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

  /////////////////////////////
  for (int i = 0; i < 4; i++)
  {
    pstModule->adwInShape[i] = pdwShape[i];
  }
  
  pstModule->pfInData = (float*)calloc(pdwShape[0]*pdwShape[1]*pdwShape[2]*pdwShape[3], sizeof(float));
  
  mx_uint out_size = 0;
  NDArrayHandle *out = 0;
  ret = MXExecutorOutputs(pstModule->hExecute, &out_size, &out);
  
  int *pdwOutShape = pstModule->adwOutShape;
  
  mx_uint out_dim = 0;
  const mx_uint *out_shape = 0;
  NDArrayHandle hout = out[0];
  ret = MXNDArrayGetShape(hout, &out_dim, &out_shape);
  assert(out_dim==2);
  size_t needsize = 1;
  for (int j = 0; j < out_dim; j++) {
    needsize *= out_shape[j];
    pdwOutShape[j] = out_shape[j];
  }
  
  for (int j = 0; j < out_size; j++)
  {
    MXNDArrayFree(out[j]);
  }
  
  assert(pdwOutShape[0]==pdwShape[0]);
  
  pstModule->pfOutScore = (float*)calloc(needsize, sizeof(float));
  pstModule->pdwClassIdx = (int*)calloc(pdwOutShape[0], sizeof(int));
  pstModule->pfClassScore = (float*)calloc(pdwOutShape[0], sizeof(float));
  
  pstModule->dwBufferSZ = 1000 * 1000 * 4;
  pstModule->pbyBuffer = (char*)calloc(pstModule->dwBufferSZ, 1);

  return 0;
}


int LPCHRECOG_Process(LPDR_HANDLE hCHRECOG, LPDR_ImageInner_S *pstImage, LPRect rect, float fStrechRatio, float fShrinkRatio, int dwStep)
{
  ModuleCHRECOG_S *pstCHRECOG = (ModuleCHRECOG_S*)hCHRECOG;
  ExecutorHandle hExecute = pstCHRECOG->hExecute;
  int dwRI, dwCI, dwPI;
  int ret = 0;
#if LPDR_TIME&0
  float costtime, diff;
  struct timeval start, end;
  
  gettimeofday(&start, NULL);
#endif
  char *pbyBufferNow = pstCHRECOG->pbyBuffer;
  int dwBufferSZNow = pstCHRECOG->dwBufferSZ;
  
  float *pfInData = pstCHRECOG->pfInData;
  float *pfOutScore = pstCHRECOG->pfOutScore;
  int dwBatchSZ = pstCHRECOG->adwInShape[0];
  int adwStdHW[2] = {pstCHRECOG->adwInShape[2], pstCHRECOG->adwInShape[3]};
  
  float *pfDataSrc = pstImage->pfData;
  int dwImgWSrc = pstImage->dwImgW;
  int dwImgHSrc = pstImage->dwImgH;
  
  int dwHCrop = rect.dwY1 - rect.dwY0 + 1;
  int dwWCrop = rect.dwX1 - rect.dwX0 + 1;
  
  int dwBlackMargin = dwHCrop / 2 + 1;
  dwWCrop += dwBlackMargin * 2;
  
  assert(dwHCrop * dwWCrop * sizeof(float) <= dwBufferSZNow);
  float *pfDataCrop = (float*)pbyBufferNow;
  pbyBufferNow += dwHCrop * dwWCrop * sizeof(float);
  dwBufferSZNow -= dwHCrop * dwWCrop * sizeof(float);
  
  memset(pfDataCrop, 0, sizeof(float) * dwHCrop * dwWCrop);
  for (dwRI = 0; dwRI < dwHCrop; dwRI++)
  {
    memcpy(pfDataCrop + dwRI * dwWCrop + dwBlackMargin, pfDataSrc + (dwRI + rect.dwY0) * dwImgWSrc + rect.dwX0, sizeof(float) * (dwWCrop - 2*dwBlackMargin));
  }
  
  int dwHDst = adwStdHW[0];
  int dwWDst = dwWCrop * dwHDst / dwHCrop;
  if (fStrechRatio > 0.f)
  {
    int dwWDst2 = dwHDst * fStrechRatio;
    if (dwWDst < dwWDst2) dwWDst = dwWDst2;
  }
  if (fShrinkRatio > 0.f) dwWDst = dwWDst * fShrinkRatio;
  
  cv::Mat imgCrop(dwHCrop, dwWCrop, CV_32FC1, pfDataCrop);
  
  assert(dwHDst * dwWDst * sizeof(float) <= dwBufferSZNow);
  float *pfDataDst = (float*)pbyBufferNow;
  pbyBufferNow += dwHDst * dwWDst * sizeof(float);
  dwBufferSZNow -= dwHDst * dwWDst * sizeof(float);
  
  cv::Mat imgDst(dwHDst, dwWDst, CV_32FC1, pfDataDst);
  
  cv::resize(imgCrop, imgDst, cv::Size(dwWDst, dwHDst), 0, 0, CV_INTER_LINEAR);
  
#if LPDR_DBG&1
  {
    cv::imshow("CHRECOG", imgDst);
    cv::waitKey(0);
  }
#endif 
  
  memset(pfInData, 0, dwBatchSZ * 1 * adwStdHW[0] * adwStdHW[1] * sizeof(float));

  int dwRealBatchNum = 0;
  for (dwPI = 0; dwPI < dwBatchSZ; dwPI++)
  {
    dwCI = dwPI * dwStep;
    if (dwCI + adwStdHW[1] >= dwWDst) break;
    float *pfInNow = pfInData + dwPI * adwStdHW[0] * adwStdHW[1];
    for (dwRI = 0; dwRI < adwStdHW[0]; dwRI++)
    {
      memcpy(pfInNow + dwRI * adwStdHW[1], pfDataDst + dwRI * dwWDst + dwCI, sizeof(float) * adwStdHW[1]);
    }
#if LPDR_DBG&0
  {
    cv::Mat img(adwStdHW[0], adwStdHW[1], CV_32FC1, pfInNow);
    cv::imshow("CHRECOG_CHR", img);
    cv::waitKey(0);
  }
#endif 
    dwRealBatchNum++;
  }
  
  NDArrayHandle hData = pstCHRECOG->args_arr[0];
  int needsize0 = getSize(pstCHRECOG->args_arr[0]);
  
  ret = MXNDArraySyncCopyFromCPU(hData, pfInData, needsize0);
  
  ret = MXExecutorForward(hExecute, 0);
  
  mx_uint out_size = 0;
  NDArrayHandle *out = 0;
  ret = MXExecutorOutputs(hExecute, &out_size, &out);

  int *pdwOutShape = pstCHRECOG->adwOutShape;
  int dwNeedSize;
  dwNeedSize = pdwOutShape[0] * pdwOutShape[1];
  
  ret = MXNDArraySyncCopyToCPU(out[0], pfOutScore, dwNeedSize);

	MXNDArrayFree(out[0]);
	
	assert(dwNeedSize * sizeof(float) <= dwBufferSZNow);
  float *pfRealOutScore = (float*)pbyBufferNow;
  pbyBufferNow += dwNeedSize * sizeof(float);
  dwBufferSZNow -= dwNeedSize * sizeof(float);
  
  memcpy(pfRealOutScore, pfOutScore, dwRealBatchNum * pdwOutShape[1] * sizeof(float));
  
  cv::Mat imgRealOut(dwRealBatchNum, pdwOutShape[1], CV_32FC1, pfRealOutScore);
  cv::Mat imgStdOut(pdwOutShape[0], pdwOutShape[1], CV_32FC1, pfOutScore);
  cv::resize(imgRealOut, imgStdOut, cv::Size(pdwOutShape[0], pdwOutShape[1]), 0, 0, CV_INTER_LINEAR);
#if LPDR_TIME&0
  gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec) / 1000.f;
	printf("CHRECOG cost:%.2fms\n", diff);
#endif
  return 0;
}


int LPCHRECOG_Release(LPDR_HANDLE hCHRECOG)
{
  int ret = 0;
  
  ModuleCHRECOG_S *pstCHRECOG = (ModuleCHRECOG_S*)hCHRECOG;
  
  ret = MXSymbolFree(pstCHRECOG->hSymbol);
	ret = MXExecutorFree(pstCHRECOG->hExecute);
	
	for (int i = 0; i < pstCHRECOG->args_num; i++) {
    ret = MXNDArrayFree(pstCHRECOG->args_arr[i]);
  }
  
  free(pstCHRECOG->pfInData);
  free(pstCHRECOG->pfOutScore);
  free(pstCHRECOG->pdwClassIdx);
  free(pstCHRECOG->pfClassScore);
  free(pstCHRECOG->pbyBuffer);
  
  free(pstCHRECOG);
  
  return 0;
}









