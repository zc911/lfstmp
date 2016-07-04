
#include "LPROIP.hpp"


int LPROIP_Create(LPDRModel_S stROIP, int dwDevType, int dwDevID, LPDR_HANDLE *phROIP)
{
  ModuleROIP_S *pstModule = (ModuleROIP_S*)calloc(1, sizeof(ModuleROIP_S));
  *phROIP = (LPDR_HANDLE)pstModule;
  
  SymbolHandle hSymbol = 0;
  int ret = 0;

  //load model
  ret = MXSymbolCreateFromJSON(stROIP.pbySym, &hSymbol);
  
  pstModule->hSymbol = hSymbol;
#if DR_DBG&0
  cout << ret << endl;
#endif

  //infer_shape
  mx_uint num_args = 2;
  const mx_uint *pdwShape = (mx_uint*)stROIP.adwShape;
  const char *keys[] = {"roip_data", "boxes"};
  const mx_uint arg_ind_ptr[] = {0, 4, 7};
  const mx_uint arg_shape_data[] = {pdwShape[0], pdwShape[1], pdwShape[2], pdwShape[3], pdwShape[4], pdwShape[5], pdwShape[6]};
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
  ret = MXNDArrayLoadFromBytes(stROIP.pbyParam, stROIP.dwParamSize, &param_size, &paramh_arr, &param_name_size, &param_names);
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

//  ret = MXExecutorForward(pstModule->hExecute, 0);
#if DR_DBG&0
 cout << ret << endl;
#endif
  //////////////////////////////////////
  pstModule->adwRectSZ[0] = pdwShape[4];
  pstModule->adwRectSZ[1] = pdwShape[5];
  pstModule->adwRectSZ[2] = pdwShape[6];
  int dwRectSZAll = pstModule->adwRectSZ[0] * pstModule->adwRectSZ[1] * pstModule->adwRectSZ[2];
  pstModule->pfRect3D = new float[dwRectSZAll];
//  cout << "roip:" << dwRectSZAll << endl;
  memset(pstModule->pfRect3D, 0, sizeof(float)*dwRectSZAll);
  
  for (int i = 0; i < 7; i++)
  {
    pstModule->adwInShape[i] = pdwShape[i];
  }
  
  mx_uint out_size = 0;
  NDArrayHandle *out = 0;
  ret = MXExecutorOutputs(pstModule->hExecute, &out_size, &out);
  assert(out_size==2);
  
  int ii = 0;
  int adwSizes[2];
  int *pdwOutShape = pstModule->adwOutShape;
  for (int i = 0; i < out_size; i++)
  {
    mx_uint out_dim = 0;
    const mx_uint *out_shape = 0;
    NDArrayHandle hout = out[i];
    ret = MXNDArrayGetShape(hout, &out_dim, &out_shape);
    size_t needsize = 1;
    for (int j = 0; j < out_dim; j++) {
      needsize *= out_shape[j];
      pdwOutShape[ii++] = out_shape[j];
//      cout << out_shape[j] << endl;
    }
    
    adwSizes[i] = needsize;
  }
  
  for (int j = 0; j < out_size; j++)
  {
    MXNDArrayFree(out[j]);
  }
  
  pstModule->pfOutCls = new float[adwSizes[0]];
  memset(pstModule->pfOutCls, 0, sizeof(float) * adwSizes[0]);
  pstModule->pfOutBB = new float[adwSizes[1]];
  memset(pstModule->pfOutBB, 0, sizeof(float)*adwSizes[1]);

  return 0;
}


int LPROIP_Process(LPDR_HANDLE hROIP, float *pfFeat4D, int adwFeatSZ[4], float *pfRect3D, int adwRectSZ[3])
{
  ModuleROIP_S *pstROIP = (ModuleROIP_S*)hROIP;
  ExecutorHandle hExecute = pstROIP->hExecute;
#if LPDR_TIME
  float costtime, diff;
  struct timeval start, end;
  
  gettimeofday(&start, NULL);
#endif
  int ret = 0;
  int needsize0 = getSize(pstROIP->args_arr[0]);
  int needsize1 = getSize(pstROIP->args_arr[1]);
  
  NDArrayHandle hRoip_Data = pstROIP->args_arr[0];
  NDArrayHandle hBoxes = pstROIP->args_arr[1];
  ret = MXNDArraySyncCopyFromCPU(hRoip_Data, pfFeat4D, needsize0);
  ret = MXNDArraySyncCopyFromCPU(hBoxes, pfRect3D, needsize1);

  ret = MXExecutorForward(hExecute, 0);

  mx_uint out_size = 0;
  NDArrayHandle *out = 0;
  ret = MXExecutorOutputs(hExecute, &out_size, &out);
//  cout << out_size << endl;
  
  int *pdwOutShape = pstROIP->adwOutShape;
  int adwNeedSizes[2];
  adwNeedSizes[0] = pdwOutShape[0] * pdwOutShape[1];
  adwNeedSizes[1] = pdwOutShape[2] * pdwOutShape[3];
  
  assert(adwRectSZ[0]*adwRectSZ[1]==pdwOutShape[0]);

  ret = MXNDArraySyncCopyToCPU(out[0], pstROIP->pfOutCls, adwNeedSizes[0]);
  ret = MXNDArraySyncCopyToCPU(out[1], pstROIP->pfOutBB, adwNeedSizes[1]);

	MXNDArrayFree(out[0]);
	MXNDArrayFree(out[1]);
#if LPDR_TIME
  gettimeofday(&end, NULL);
	diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
	printf("ROIP cost:%.2fms\n", diff);
#endif
  return 0;
}


int LPROIP_Release(LPDR_HANDLE hROIP)
{
  int ret = 0;
  
  ModuleROIP_S *pstROIP = (ModuleROIP_S*)hROIP;
  
  ret = MXSymbolFree(pstROIP->hSymbol);
	ret = MXExecutorFree(pstROIP->hExecute);
	
	for (int i = 0; i < pstROIP->args_num; i++) {
    ret = MXNDArrayFree(pstROIP->args_arr[i]);
  }
  
  delete []pstROIP->pfOutCls;
  delete []pstROIP->pfOutBB;
  delete []pstROIP->pfRect3D;
  
  free(pstROIP);
  
  return 0;
}














