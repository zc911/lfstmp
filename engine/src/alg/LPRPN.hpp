
#ifndef __LP_RPN_HPP__
#define __LP_RPN_HPP__


#include "LPDetectRecog.hpp"
#include "LPInner.hpp"


struct ModuleRPN_S {
	ExecutorHandle hExecute;
	NDArrayHandle *args_arr;
	int args_num = 0; 
	SymbolHandle hSymbol;

	int adwInShape[5]; //input image shape
	int adwOutShape[4+4+4]; //anchor class map shape, anchor bb map shape, feat shape
	
	uchar *pubyInputData;
	float *pfInputData;
	
	float *pfOutputDataFeat;
	float *pfOutputDataCls;
	float *pfOutputDataReg;
	
	uchar *pubyBuffer;
	int dwBuffSize;
	
//	vector<LPRectInfo> lprects;
  int dwGroupSize;
	vector<LPRectInfo> *plprectgroup;
	vector<LPRectInfo> *plprectgroup_0;
	
  int	*pdwRealWs;
  int	*pdwRealHs;
};


int LPRPN_Create(LPDRModel_S stRPN, int dwDevType, int dwDevID, LPDR_HANDLE *phRPN);

int LPRPN_Process(LPDR_HANDLE hRPN, LPDR_ImageInner_S *pstImgSet, int dwImgNum);

int LPRPN_Release(LPDR_HANDLE hRPN);


#endif


