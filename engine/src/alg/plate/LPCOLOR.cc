
#include "LPCOLOR.hpp"


int LPCOLOR_Create(LPDRModel_S stCOLOR, int dwDevType, int dwDevID, LPDR_HANDLE *phCOLOR)
{
  ModuleCOLOR_S *pstModule = (ModuleCOLOR_S*)calloc(1, sizeof(ModuleCOLOR_S));
  *phCOLOR = (LPDR_HANDLE)pstModule;
  
  SymbolHandle hSymbol = 0;
  int ret = 0;

  //load model
  ret = MXSymbolCreateFromJSON(stCOLOR.pbySym, &hSymbol);
  
  pstModule->hSymbol = hSymbol;
  assert(ret==0);
#if DR_DBG&0
  cout << ret << endl;
#endif

  //infer_shape
  mx_uint num_args = 1;
  const mx_uint *pdwShape = (mx_uint*)stCOLOR.adwShape;
//  cout << pdwShape[0] << "," << pdwShape[1] << "," << pdwShape[2] << "," << pdwShape[3] << "\n";
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
  ret = MXNDArrayLoadFromBytes(stCOLOR.pbyParam, stCOLOR.dwParamSize, &param_size, &paramh_arr, &param_name_size, &param_names);
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
  
  pstModule->dwBufferSZ = 1000 * 1000 * 4;
  pstModule->pbyBuffer = (char*)calloc(pstModule->dwBufferSZ, 1);

  return 0;
}


int LPCOLOR_Process(LPDR_HANDLE hCOLOR, float *pfData, int dwDH, int dwDW, int *pdwColor)
{
  ModuleCOLOR_S *pstCOLOR = (ModuleCOLOR_S*)hCOLOR;
  ExecutorHandle hExecute = pstCOLOR->hExecute;
  int dwRI, dwCI, dwPI;
  int ret = 0;
#if LPDR_TIME&0
  float costtime, diff;
  struct timeval start, end;
  
  gettimeofday(&start, NULL);
#endif
  char *pbyBufferNow = pstCOLOR->pbyBuffer;
  int dwBufferSZNow = pstCOLOR->dwBufferSZ;
  
  float *pfInData = pstCOLOR->pfInData;
  float *pfOutScore = pstCOLOR->pfOutScore;
  int dwBatchSZ = pstCOLOR->adwInShape[0];
  int adwStdHW[2] = {pstCOLOR->adwInShape[2], pstCOLOR->adwInShape[3]};
//  cout << adwStdHW[0] << "," << adwStdHW[1] << "\n";
  cv::Mat imgOri(dwDH, dwDW, CV_32FC3, pfData);
  cv::Mat imgDst(adwStdHW[0], adwStdHW[1], CV_32FC3, pfInData);
  
  cv::resize(imgOri, imgDst, cv::Size(adwStdHW[1], adwStdHW[0]), 0, 0, CV_INTER_LINEAR);

#if LPDR_DBG
  {
    cv::imshow("COLOR", imgDst);
    cv::waitKey(10);
  }
#endif 

  NDArrayHandle hData = pstCOLOR->args_arr[0];
  int needsize0 = getSize(pstCOLOR->args_arr[0]);
  
  ret = MXNDArraySyncCopyFromCPU(hData, pfInData, needsize0);
  
  ret = MXExecutorForward(hExecute, 0);
  
  mx_uint out_size = 0;
  NDArrayHandle *out = 0;
  ret = MXExecutorOutputs(hExecute, &out_size, &out);

  int *pdwOutShape = pstCOLOR->adwOutShape;
  int dwNeedSize;
  dwNeedSize = pdwOutShape[0] * pdwOutShape[1];
  
  ret = MXNDArraySyncCopyToCPU(out[0], pfOutScore, dwNeedSize);

	MXNDArrayFree(out[0]);
	
	int dwClassIdx = 0;
	float fClassScore = 0.f;
	for (dwPI = 0; dwPI < dwNeedSize; dwPI++)
	{
//	  cout << dwPI << ":" << pfOutScore[dwPI] << ", ";
	  if (fClassScore < pfOutScore[dwPI])
	  {
	    fClassScore = pfOutScore[dwPI];
	    dwClassIdx = dwPI;
	  }
	}
//	cout << endl;
	if (dwClassIdx > LP_COLOUR_YELLOW && fClassScore < 0.80f && (pfOutScore[LP_COLOUR_BLUE] > 0.10f || pfOutScore[LP_COLOUR_YELLOW] > 0.10f))
	{
	  if (pfOutScore[LP_COLOUR_BLUE] > pfOutScore[LP_COLOUR_YELLOW])
	  {
	    fClassScore = pfOutScore[LP_COLOUR_BLUE];
	    dwClassIdx = LP_COLOUR_BLUE;
	  }
	  else
	  {
	    fClassScore = pfOutScore[LP_COLOUR_YELLOW];
	    dwClassIdx = LP_COLOUR_YELLOW;
	  }
	}
	
	*pdwColor = dwClassIdx;

#if LPDR_TIME&0
  gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec) / 1000.f;
	printf("COLOR cost:%.2fms\n", diff);
#endif
  return 0;
}


int LPCOLOR_Release(LPDR_HANDLE hCOLOR)
{
  int ret = 0;
  
  ModuleCOLOR_S *pstCOLOR = (ModuleCOLOR_S*)hCOLOR;
  
  ret = MXSymbolFree(pstCOLOR->hSymbol);
	ret = MXExecutorFree(pstCOLOR->hExecute);
	
	for (int i = 0; i < pstCOLOR->args_num; i++) {
    ret = MXNDArrayFree(pstCOLOR->args_arr[i]);
  }
  
  free(pstCOLOR->pfInData);
  free(pstCOLOR->pfOutScore);
  free(pstCOLOR->pbyBuffer);
  
  free(pstCOLOR);
  
  return 0;
}


/*
#define LP_COLOUR_UNKNOWN   0
#define LP_COLOUR_BLUE      1
#define LP_COLOUR_YELLOW    2
#define LP_COLOUR_WHITE     3
#define LP_COLOUR_BLACK     4
#define LP_COLOUR_GREEN     5
*/
int doRecogColors_NN(LPDR_HANDLE hCOLOR, LPDR_ImageSet_S *pstImgSet, LPDR_OutputSet_S *pstOutputSet)
{
  int dwSI, dwLI, dwRI, dwCI;
  int dwImgNum = pstImgSet->dwImageNum;
  int dwImgW, dwImgH;
  uchar *pubyImgData, *pubyRow;
  float *pfRowB, *pfRowG, *pfRowR;
  uchar *pubyPlateTmp = 0;
  int dwLPNum = 0, dwMaxColor;
  LPDRInfo_S *pstLPDR = 0;
  int adwBB[4];
  
  float *pfPlateTmp = new float[1000*1000];
  pubyPlateTmp = new uchar[1000*1000];
  
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
      adwBB[0] = pstLPDR->adwLPRect[0] + dwBBW/8;
      adwBB[1] = pstLPDR->adwLPRect[1] + dwBBH/8;
      adwBB[2] = pstLPDR->adwLPRect[2] - dwBBW/8;
      adwBB[3] = pstLPDR->adwLPRect[3] - dwBBH/8;
      
      dwBBW = adwBB[2] - adwBB[0] + 1;
      dwBBH = adwBB[3] - adwBB[1] + 1;
      
      if (dwBBH * dwBBW * 3 > 1000*1000) continue;
#if 1
      for (dwRI = adwBB[1]; dwRI <= adwBB[3]; dwRI++)
      {
        pubyRow = pubyImgData + dwRI * 3 * dwImgW;
        pfRowB = pfPlateTmp + (dwRI - adwBB[1]) * dwBBW;
        pfRowG = pfRowB + dwBBH * dwBBW;
        pfRowR = pfRowG + dwBBH * dwBBW;
        for (dwCI = adwBB[0]; dwCI <= adwBB[2]; dwCI++)
        {
          pfRowB[dwCI - adwBB[0]] = pubyRow[dwCI * 3 + 0] / 255.f;
          pfRowG[dwCI - adwBB[0]] = pubyRow[dwCI * 3 + 1] / 255.f;
          pfRowR[dwCI - adwBB[0]] = pubyRow[dwCI * 3 + 2] / 255.f;
        }
      }
#else
      for (dwRI = adwBB[1]; dwRI <= adwBB[3]; dwRI++)
      {
        pubyRow = pubyImgData + dwRI * 3 * dwImgW;
        memcpy(pubyPlateTmp + (dwRI - adwBB[1]) * dwBBW * 3, pubyRow + adwBB[0] * 3, dwBBW * 3);
      }
      
      cv::Mat matubyPlate(dwBBH, dwBBW, CV_8UC3, pubyPlateTmp);
      cv::Mat matfPlate(dwBBH, dwBBW, CV_32FC3, pfPlateTmp);
      matubyPlate.convertTo(matfPlate, CV_32FC3, 1.0f/255.f, 0);
#if LPDR_DBG&1
  cv::namedWindow("matfPlate", 0);
  cv::imshow("matfPlate", matfPlate);
  cv::waitKey(10);
#endif
#endif
      dwMaxColor = LP_COLOUR_UNKNOWN;
      LPCOLOR_Process(hCOLOR, pfPlateTmp, dwBBH, dwBBW, &dwMaxColor);
//      printf("dwMaxColor:%d\n", dwMaxColor);
      
      pstLPDR->dwColor = dwMaxColor;
    }

  }
  
  delete []pfPlateTmp;
  delete []pubyPlateTmp;
  
  return 0;
}






