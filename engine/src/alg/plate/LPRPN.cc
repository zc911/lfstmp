
#include "LPRPN.hpp"


int gdwAnchorBoxes[ANCHOR_NUM][2] = {{60, 300}, {60, 210}, {60, 146}, {60, 102}, {60, 72}, {51, 258}, {51, 180}, {51, 126}, {51, 88}, {51, 61}, {44, 221}, {44, 155}, {44, 108}, {44, 76}, {44, 53}, {38, 190}, {38, 133}, {38, 93}, {38, 65}, {38, 45}, {32, 164}, {32, 114}, {32, 80}, {32, 56}, {28, 141}, {28, 98}, {28, 69}, {28, 48}, {24, 121}, {24, 84}, {24, 59}, {20, 104}, {20, 73}, {20, 51}};


int getRPN_Rects(int imgh, int imgw, int adims[2], int ashapes[2][4],
			mx_float *pfScore, int scoreSize, mx_float *pfRect, int rectSize, vector<LPRectInfo> &lprects);


int LPRPN_Create(LPDRModel_S stRPN, int dwDevType, int dwDevID, LPDR_HANDLE *phRPN)
{
  ModuleRPN_S *pstModule = (ModuleRPN_S*)calloc(1, sizeof(ModuleRPN_S));
  *phRPN = (LPDR_HANDLE)pstModule;
  
  SymbolHandle hSymbol = 0;
  int ret = 0;

  //load model
  ret = MXSymbolCreateFromJSON(stRPN.pbySym, &hSymbol);
  
  pstModule->hSymbol = hSymbol;
  assert(ret==0);
#if DR_DBG&0
  cout << ret << endl;
#endif

  //infer_shape

  mx_uint num_args = 1;
  const mx_uint *pdwShape = (mx_uint*)stRPN.adwShape;
  const mx_uint dwStdH = pdwShape[3];
  const mx_uint dwStdW = pdwShape[4];
  const mx_uint dwBatchSize = pdwShape[0] * pdwShape[1];
  const char* keys = {"data"};
  const mx_uint arg_ind_ptr[2] = {0, 4};
  const mx_uint arg_shape_data[4] = {dwBatchSize, pdwShape[2], pdwShape[3], pdwShape[4]};
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

  ret = MXSymbolInferShape(hSymbol, num_args, &keys, arg_ind_ptr, arg_shape_data,
              &in_shape_size, &in_shape_ndim, &in_shape_data,
              &out_shape_size, &out_shape_ndim, &out_shape_data,
              &aux_shape_size, &aux_shape_ndim, &aux_shape_data, &complete);

#if DR_DBG
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


  //load parameters from trained file.
  unordered_map<string, NDArrayHandle> param_pairs;
  mx_uint param_size = 0;
  NDArrayHandle* paramh_arr = 0;
  mx_uint param_name_size = 0;
  const char** param_names = 0;
  ret = MXNDArrayLoadFromBytes(stRPN.pbyParam, stRPN.dwParamSize, &param_size, &paramh_arr, &param_name_size, &param_names);
  assert(param_size == param_name_size);
  for (int i = 0 ; i < param_name_size; i++) {
#if DR_DBG
    cout << i << ":" << param_names[i] << endl;
#endif
    param_pairs[param_names[i]] = paramh_arr[i];
  }

  //list arguments
  mx_uint args_size = 0;
  const char **args_str_array = 0;
  ret = MXSymbolListArguments(hSymbol, &args_size, &args_str_array);
#if DR_DBG
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
  
  unordered_map<string, NDArrayHandle> allparams_pairs;
  for (int i = 0; i < in_shape_size; i++) {
    ret = MXNDArrayCreate(in_shape_data[i], in_shape_ndim[i], dwDevType, dwDevID, 0, &in_args[i]);
    allparams_pairs[args_str_array[i]] = in_args[i];
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
#if DR_DBG
   cout << i << ":" << name << "--" << allparams_pairs[name];
   cout << ", (";
   for (int j = 0; j < in_shape_ndim[i]; j++)
   {
    cout << in_shape_data[i][j] << ", ";
   }
   cout << ")";
   cout << endl;
#endif
   if (param_pairs[name]) {
     NDArrayHandle hparam = param_pairs[name];
     NDArrayHandle hparamto = allparams_pairs[name];
     copy_ndarray(hparam, hparamto, pfBuffer, maxbuffer_size);
   }
  }
  
  //copy pprpn_anchorinfo and pprpn_otherinfo
  NDArrayHandle hparam0 = allparams_pairs["pprpn_anchorinfo"];
  for (int dwAI = 0; dwAI < ANCHOR_NUM; dwAI++)
  {
    pfBuffer[dwAI * 2 + 0] = gdwAnchorBoxes[dwAI][0];
    pfBuffer[dwAI * 2 + 1] = gdwAnchorBoxes[dwAI][1];
  }
  
  MXNDArraySyncCopyFromCPU(hparam0, pfBuffer, ANCHOR_NUM*2);
//  cout << "3\n";
  NDArrayHandle hparam1 = allparams_pairs["pprpn_otherinfo"];
  pfBuffer[0] = LP_SCORE_MAX;
  pfBuffer[1] = dwStdH;
  pfBuffer[2] = dwStdW;
  MXNDArraySyncCopyFromCPU(hparam1, pfBuffer, 3);
//  cout << "4\n";
  free(pfBuffer);
  //free loaded parameters ndarray
  //cout << args_size << ", " << param_size << endl;
  for (int i = 0; i < param_size; i++) {
    MXNDArrayFree(paramh_arr[i]);
  }

//  cout << "5\n";
 //do bind
  ret = MXExecutorBind(hSymbol, dwDevType, dwDevID,
             len, in_args, arg_grad_store,
             grad_req_type, aux_states_len,
             aux_states, &pstModule->hExecute);

  pstModule->args_arr = in_args;
  pstModule->args_num = args_size;
  
//  cout << "6\n";
  ret = MXExecutorForward(pstModule->hExecute, 0);
  
//  cout << "7\n";
#if DR_DBG&0
 cout << ret << endl;
#endif

  /////////////////////////////////////////////////////////////
  for (int i = 0; i < 5; i++)
  {
    pstModule->adwInShape[i] = stRPN.adwShape[i];
  }
  
  int *pdwInShape = pstModule->adwInShape;
  int dwInputSize = pdwInShape[0] * pdwInShape[1] * pdwInShape[2] * pdwInShape[3] * pdwInShape[4];
  
  pstModule->pubyInputData = (uchar*)calloc(dwInputSize, 1);
  pstModule->pfInputData = (float*)calloc(dwInputSize, sizeof(float));

  pstModule->dwBuffSize = 1024*1024*1024;
  pstModule->pubyBuffer = (uchar*)calloc(pstModule->dwBuffSize, 1);
  
  mx_uint out_size = 0;
  NDArrayHandle *out = 0;
  ret = MXExecutorOutputs(pstModule->hExecute, &out_size, &out);
  assert(out_size==2);
  int *pdwOutShape = pstModule->adwOutShape;
  
  int adwSizes[2];
  int ii = 0;
  for (int i = 0; i < 2; i++)
  {
    mx_uint out_dim = 0;
    const mx_uint *out_shape = 0;
    NDArrayHandle hout = out[i];
    ret = MXNDArrayGetShape(hout, &out_dim, &out_shape);
    size_t needsize = 1;
    for (int j = 0; j < out_dim; j++) {
      needsize *= out_shape[j];
      pdwOutShape[ii++] = out_shape[j];
//      cout << out_shape[j] << ", ";
    }
//    cout << endl;
    
    adwSizes[i] = needsize;
  }
  
  for (int j = 0; j < out_size; j++)
  {
    MXNDArrayFree(out[j]);
  }

  pstModule->pfOutputBBs = (float*)calloc(adwSizes[0], sizeof(float));
  pstModule->pfOutputDataFeat = (float*)calloc(adwSizes[1], sizeof(float));
  
  pstModule->dwGroupSize = dwBatchSize;
  pstModule->plprectgroup = new vector<LPRectInfo>[pstModule->dwGroupSize];
  pstModule->plprectgroup_0 = new vector<LPRectInfo>[pstModule->dwGroupSize];
  
  
  pstModule->pdwRealWs = new int[dwBatchSize];
  memset(pstModule->pdwRealWs, 0, sizeof(int) * dwBatchSize);
  pstModule->pdwRealHs = new int[dwBatchSize];
  memset(pstModule->pdwRealHs, 0, sizeof(int) * dwBatchSize);
  
//  pstModule->adwRectSZ[0] = dwBatchSize;
//  pstModule->adwRectSZ[0] = dwBatchSize;
//  pstModule->pfRect3D = new float[stRPN.adwShape[]];
//	int adwRectSZ[3];

  return 0;
}


int LPRPN_Process(LPDR_HANDLE hRPN, LPDR_ImageInner_S *pstImgSet, int dwImgNum)
{
  ModuleRPN_S *pstRPN = (ModuleRPN_S*)hRPN;
  ExecutorHandle hExecute = pstRPN->hExecute;
#if LPDR_TIME
  float costtime, diff;
  struct timeval start, end;
  
  gettimeofday(&start, NULL);
#endif

//  cout << "fuck....\n";

  int ret = 0;
  int needsize = getSize(pstRPN->args_arr[0]);
  float *pfInputData = pstRPN->pfInputData;
  int *pdwInShape = pstRPN->adwInShape;
  int dwStdW = pdwInShape[4];
  int dwStdH = pdwInShape[3];
  int dwBatchSz = pdwInShape[0] * pdwInShape[1];
  int dwInputOneSize = pdwInShape[2] * pdwInShape[3] * pdwInShape[4];
  NDArrayHandle hData = pstRPN->args_arr[0];
  
  assert(dwBatchSz==dwImgNum);

  assert(needsize==pdwInShape[0]*pdwInShape[1]*pdwInShape[2]*pdwInShape[3]*pdwInShape[4]);

  memset(pfInputData, 0, sizeof(float) * needsize);

  int *pdwRealWs = pstRPN->pdwRealWs;
  memset(pdwRealWs, 0, sizeof(int) * dwImgNum);
  int *pdwRealHs = pstRPN->pdwRealHs;
  memset(pdwRealHs, 0, sizeof(int) * dwImgNum);
  for (int i = 0; i < dwImgNum; i++)
  {
    float *pfData = pstImgSet[i].pfData;
    int dwImgWOri = pstImgSet[i].dwImgW;
    int dwImgHOri = pstImgSet[i].dwImgH;
    if (pfData)
    {
      int dwRealW = 0, dwRealH = 0;
      float *pfInputDataOne = pfInputData + dwInputOneSize * i;
//      imgResizeAddBlack_f(pfData, dwImgWOri, dwImgHOri, pfInputDataOne, 
//                          dwStdW, dwStdH, &dwRealW, &dwRealH);
      imgResizeAddBlack_fNorm(pfData, dwImgWOri, dwImgHOri, pfInputDataOne, 
                          dwStdW, dwStdH, &dwRealW, &dwRealH);
      pdwRealWs[i] = dwRealW;
      pdwRealHs[i] = dwRealH;

#if LPDR_DBG&0
        cv::Mat gimg(dwStdH, dwStdW, CV_32FC1, pfInputDataOne);
//        cv::namedWindow("rpninput", 0);
        cv::imshow("rpninput", gimg);
        cv::waitKey(0);
#endif
    }
  }
#if LPDR_TIME
  gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
	printf("RPN cost_0[%d]:%.2fms\n", dwImgNum, diff);
#endif
  int *pdwOutShape = pstRPN->adwOutShape;
  int adwNeedSizes[2];
  int dwBoxNumPerBatch = 0;
  int dwBoxDim = 0;
  
  

  for (int loopi = 0; loopi < 1; loopi++)
  {
  #if LPDR_TIME
	  gettimeofday(&start, NULL);
  #endif

    ret = MXNDArraySyncCopyFromCPU(hData, pfInputData, needsize);

    ret = MXExecutorForward(hExecute, 0);

    mx_uint out_size = 0;
    NDArrayHandle *out = 0;

    ret = MXExecutorOutputs(hExecute, &out_size, &out);

  //  MXNDArrayWaitAll();
    for (int wi = 0; wi < out_size; wi++)
    {
      MXNDArrayWaitToRead(out[wi]);
    }
    
    dwBoxNumPerBatch = pdwOutShape[1];
    dwBoxDim = pdwOutShape[2];
    adwNeedSizes[0] = pdwOutShape[0] * dwBoxNumPerBatch * dwBoxDim;
    adwNeedSizes[1] = pdwOutShape[3] * pdwOutShape[4] * pdwOutShape[5] * pdwOutShape[6];

    ret = MXNDArraySyncCopyToCPU(out[0], pstRPN->pfOutputBBs, adwNeedSizes[0]);
    ret = MXNDArraySyncCopyToCPU(out[1], pstRPN->pfOutputDataFeat, adwNeedSizes[1]);

	  MXNDArrayFree(out[0]);
	  MXNDArrayFree(out[1]);
	
#if LPDR_TIME
    gettimeofday(&end, NULL);
	  diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
	  printf("RPN cost_1[%d]:%.2fms\n", dwImgNum, diff);
	
	  gettimeofday(&start, NULL);
  #endif
  }


  for (int dwBI = 0; dwBI < pstRPN->dwGroupSize; dwBI++)
  {
    pstRPN->plprectgroup[dwBI].clear();
    pstRPN->plprectgroup_0[dwBI].clear();
  }

  for (int dwBI = 0; dwBI < dwBatchSz; dwBI++)
  {
    float *pfData = pstImgSet[dwBI].pfData;
    int *pdwPRect = pstImgSet[dwBI].adwPRect;
    int dwRealW = pdwRealWs[dwBI];
    int dwRealH = pdwRealHs[dwBI];
    int dwImgWOri = pstImgSet[dwBI].dwImgW;
    int dwImgHOri = pstImgSet[dwBI].dwImgH;
//    int dwPID = pstImgSet[dwBI].dwPID;
    float *pfRectsBatchNow = pstRPN->pfOutputBBs + dwBI * dwBoxNumPerBatch * dwBoxDim;
    
    int dwX0, dwY0, dwX1, dwY1;
    
    if (pfData)
    {
//      for (int dwCI = 0; dwCI < lprects0.size(); dwCI++)
      for (int dwCI = 0; dwCI < dwBoxNumPerBatch; dwCI++)
      {
        float *pfRectNow = pfRectsBatchNow + dwCI * dwBoxDim;
        if (pfRectNow[2] > 0.f)
        {
          dwX0 = pfRectNow[1] - pfRectNow[3] / 2;
          dwY0 = pfRectNow[0] - pfRectNow[2] / 2;
          dwX1 = pfRectNow[1] + pfRectNow[3] / 2;
          dwY1 = pfRectNow[0] + pfRectNow[2] / 2;
          
          dwX0 = max(0, dwX0);
          dwY0 = max(0, dwY0);
          dwX1 = min(dwRealW - 1, dwX1);
          dwY1 = min(dwRealH - 1, dwY1);
          
          if (dwX0 >= dwX1 || dwY0 >= dwY1) continue;
          
          LPRectInfo rect(1.0f, (dwY0 + dwY1) / 2, (dwX0 + dwX1) / 2, dwY1 - dwY0 + 1, dwX1 - dwX0 + 1);
          
          pstRPN->plprectgroup_0[dwBI].push_back(rect);
          
          rect.fWidth = rect.fWidth * dwImgWOri / dwRealW;
          rect.fHeight = rect.fHeight * dwImgHOri / dwRealH;
          rect.fCentX = rect.fCentX * dwImgWOri / dwRealW;
          rect.fCentY = rect.fCentY * dwImgHOri / dwRealH;
          rect.fCentX += pdwPRect[0];
          rect.fCentY += pdwPRect[1];
          
          pstRPN->plprectgroup[dwBI].push_back(rect);
        }
      }
    }
#if LPDR_DBG
    cv::Mat gimg2(dwStdH, dwStdW, CV_32FC1, pfInputData + dwBI * dwStdH * dwStdW);
    cv::Mat cimg2(dwStdH, dwStdW, CV_32FC3);
    cv::cvtColor(gimg2, cimg2, CV_GRAY2BGR);
    auto &lproipnms_one = pstRPN->plprectgroup_0[dwBI];
    for (int dwRI = 0; dwRI < lproipnms_one.size(); dwRI++)
    {
      LPRectInfo &lprect = lproipnms_one[dwRI];
        
      int dwX0 = lprect.fCentX - lprect.fWidth/2;
      int dwY0 = lprect.fCentY - lprect.fHeight/2;
      int dwX1 = lprect.fCentX + lprect.fWidth/2;
      int dwY1 = lprect.fCentY + lprect.fHeight/2;
      
      cv::rectangle(cimg2, cv::Point(dwX0, dwY0), cv::Point(dwX1, dwY1), CV_RGB(255, 0, 0), 1, 8, 0);
    }
    cv::imshow("rpnoutput", cimg2);
    cv::waitKey(0);
#endif
  }

#if LPDR_TIME
  gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
	printf("RPN cost_2:%.2fms\n", diff);
#endif
  return 0;
}


int LPRPN_Release(LPDR_HANDLE hRPN)
{
  int ret = 0;
  
  ModuleRPN_S *pstRPN = (ModuleRPN_S*)hRPN;
  
  ret = MXSymbolFree(pstRPN->hSymbol);
	ret = MXExecutorFree(pstRPN->hExecute);
	
	for (int i = 0; i < pstRPN->args_num; i++) {
    ret = MXNDArrayFree(pstRPN->args_arr[i]);
  }
  
  free(pstRPN->pubyInputData);
  free(pstRPN->pfInputData);
  free(pstRPN->pubyBuffer);
  
  free(pstRPN->pfOutputBBs);
  free(pstRPN->pfOutputDataFeat);
  
  delete []pstRPN->plprectgroup;
  delete []pstRPN->plprectgroup_0;
  delete []pstRPN->pdwRealWs;
  delete []pstRPN->pdwRealHs;
  
  free(pstRPN);
  
  return 0;
}


int getRPN_Rects(int imgh, int imgw, int adims[2], int ashapes[2][4],
			mx_float *pfScore, int scoreSize, mx_float *pfRect, int rectSize, vector<LPRectInfo> &lprects)
{
	int anchorNum = ashapes[0][1];
	int height = ashapes[0][2];
	int width = ashapes[0][3];
	float afRect[4];

	assert(anchorNum == ANCHOR_NUM);
	assert(ANCHOR_NUM * 4 == ashapes[1][1]);
	assert(height == ashapes[1][2]);
	assert(width == ashapes[1][3]);

	mx_float *pfAnchorScore = 0;
	mx_float *pfAnchorRect = 0;
	mx_float *pfCentX = 0, *pfCentY = 0, *pfHeight = 0, *pfWidth = 0;
	int dwOft = 0;

	for (int ai = 0; ai < ANCHOR_NUM; ai++) {
		int anchorH = gdwAnchorBoxes[ai][0];
		int anchorW = gdwAnchorBoxes[ai][1];
		pfAnchorScore = pfScore + ai * height * width;
		pfAnchorRect = pfRect + ai * height * width * 4;
		pfCentY = pfAnchorRect;
		pfCentX = pfCentY + height * width;
		pfHeight = pfCentX + height * width;
		pfWidth = pfHeight + height * width;
		for (int hi = 0; hi < height; hi++) {
			for (int wi = 0; wi < width; wi++) {
				dwOft = hi * width + wi;
				if (pfAnchorScore[dwOft] > LP_SCORE_MAX) {
					afRect[0] = pfCentY[dwOft] * anchorH + hi * imgh / height;
					afRect[1] = pfCentX[dwOft] * anchorW + wi * imgw / width;
					afRect[2] = expf(pfHeight[dwOft]) * anchorH;
					afRect[3] = expf(pfWidth[dwOft]) * anchorW;
					lprects.push_back(LPRectInfo(pfAnchorScore[dwOft],
																				afRect[0], afRect[1],
																				afRect[2], afRect[3]));
				}
			}
		}
	}
	
	return 0;
}


