
#include "LPPREG.hpp"

int LPPREG_Create(LPDRModel_S stPREG, int dwDevType, int dwDevID, LPDR_HANDLE *phPREG)
{
  ModulePREG_S *pstModule = (ModulePREG_S*)calloc(1, sizeof(ModulePREG_S));
  *phPREG = (LPDR_HANDLE)pstModule;
  
  SymbolHandle hSymbol = 0;
  int ret = 0;

  //load model
  ret = MXSymbolCreateFromJSON(stPREG.pbySym, &hSymbol);
  
  pstModule->hSymbol = hSymbol;
  assert(ret==0);
#if DR_DBG&0
  cout << ret << endl;
#endif

  //infer_shape
  mx_uint num_args = 1;
  const mx_uint *pdwShape = (mx_uint*)stPREG.adwShape;
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
  ret = MXNDArrayLoadFromBytes(stPREG.pbyParam, stPREG.dwParamSize, &param_size, &paramh_arr, &param_name_size, &param_names);
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

  ///////////////////////////////
  for (int j = 0; j < 4; j++)
  {
    pstModule->adwInShape[j] = pdwShape[j];
  }
  
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
  
  assert(pdwShape[0]==1);
  
  pstModule->pfStdData = new float[pdwShape[0] * pdwShape[1] * pdwShape[2] * pdwShape[3]];
  pstModule->pfOutPolygon = new float[needsize];

  return 0;
}


int LPPREG_Process(LPDR_HANDLE hPREG, LPDR_ImageInner_S *pstImage, int adwPolygonOut[12])
{
  ModulePREG_S *pstPREG = (ModulePREG_S*)hPREG;
  ExecutorHandle hExecute = pstPREG->hExecute;
#if LPDR_TIME
  float costtime, diff;
  struct timeval start, end;

  gettimeofday(&start, NULL);
#endif
//  cout << "LPPREG_Process 0\n";
  int ret = 0;
  int needsize0 = getSize(pstPREG->args_arr[0]);
  float *pfStdData = pstPREG->pfStdData;
  float *pfSrcData = pstImage->pfData;
  int dwSrcW = pstImage->dwImgW;
  int dwSrcH = pstImage->dwImgH;
  int dwStdW = pstPREG->adwInShape[3];
  int dwStdH = pstPREG->adwInShape[2];
  float *pfPolygon = pstPREG->pfOutPolygon;
//  cout << "LPPREG_Process 1\n";
  cv::Mat srcImg(dwSrcH, dwSrcW, CV_32FC1, pfSrcData);
  cv::Mat tmpImg(dwStdH, dwStdW, CV_32FC1, pfStdData);
	cv::resize(srcImg, tmpImg, cv::Size(dwStdW, dwStdH), 0, 0, CV_INTER_LINEAR);
//	memcpy(pfStdData, (float*)tmpImg.data, sizeof(float)*needsize0);
  
#if LPDR_DBG&0
  cout << "H:" << dwStdH << "W:" << dwStdW << endl;
  cv::Mat gimg(dwStdH, dwStdW, CV_32FC1, pfStdData);
  cv::Mat cimg(dwStdH, dwStdW, CV_32FC3);
  cv::cvtColor(gimg, cimg, CV_GRAY2BGR);
  cv::imshow("LPPREG_Process", cimg);
  cv::waitKey(0);
#endif

  NDArrayHandle hPReg_Data = pstPREG->args_arr[0];
  
  ret = MXNDArraySyncCopyFromCPU(hPReg_Data, pfStdData, needsize0);

  ret = MXExecutorForward(hExecute, 0);

  mx_uint out_size = 0;
  NDArrayHandle *out = 0;
  ret = MXExecutorOutputs(hExecute, &out_size, &out);

  int *pdwOutShape = pstPREG->adwOutShape;
  int dwNeedSize;
  dwNeedSize = pdwOutShape[0] * pdwOutShape[1];

  ret = MXNDArraySyncCopyToCPU(out[0], pfPolygon, dwNeedSize);

	MXNDArrayFree(out[0]);

	adwPolygonOut[0] = (int)(pfPolygon[0] * dwSrcW);
	adwPolygonOut[1] = (int)(pfPolygon[1] * dwSrcH);
	adwPolygonOut[2] = (int)(pfPolygon[2] * dwSrcW);
	adwPolygonOut[3] = (int)(pfPolygon[3] * dwSrcH);
	adwPolygonOut[4] = (int)(pfPolygon[4] * dwSrcW);
	adwPolygonOut[5] = (int)(pfPolygon[5] * dwSrcH);
	adwPolygonOut[6] = (int)(pfPolygon[6] * dwSrcW);
	adwPolygonOut[7] = (int)(pfPolygon[7] * dwSrcH);
	
	adwPolygonOut[8] = (int)(pfPolygon[8] * dwSrcW + 0.5f);
	adwPolygonOut[9] = (int)(pfPolygon[9] * dwSrcH + 0.5f);
	adwPolygonOut[10] = (int)(pfPolygon[10] * dwSrcW + 0.5f);
	adwPolygonOut[11] = (int)(pfPolygon[11] * dwSrcH + 0.5f);
#if LPDR_TIME
  gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec) / 1000.f;
	printf("PREG cost:%.2fms\n", diff);
#endif
  return 0;
}


int LPPREG_Release(LPDR_HANDLE hPREG)
{
  int ret = 0;
  
  ModulePREG_S *pstPREG = (ModulePREG_S*)hPREG;
  
  ret = MXSymbolFree(pstPREG->hSymbol);
	ret = MXExecutorFree(pstPREG->hExecute);
	
	for (int i = 0; i < pstPREG->args_num; i++) {
    ret = MXNDArrayFree(pstPREG->args_arr[i]);
  }
  
  delete []pstPREG->pfStdData;
  delete []pstPREG->pfOutPolygon;
  
  free(pstPREG);

  return 0;
}



