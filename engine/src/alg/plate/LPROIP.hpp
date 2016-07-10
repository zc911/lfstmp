
#ifndef __LP_ROIP_HPP__
#define __LP_ROIP_HPP__


#include "LPDetectRecog.hpp"
#include  "LPInner.hpp"


struct ModuleROIP_S {
	ExecutorHandle hExecute;
	NDArrayHandle *args_arr;
	int args_num = 0; 
	SymbolHandle hSymbol;

	int adwInShape[4+3]; //input feat shape, box shape
	int adwOutShape[2+2]; //class shape, bb shape
	
  float *pfRect3D;
	int adwRectSZ[3];
	
	float *pfOutCls;
	float *pfOutBB;
	
	vector<LPRectInfo> *pvBBGroup;
};


int LPROIP_Create(LPDRModel_S stROIP, int dwDevType, int dwDevID, LPDR_HANDLE *phROIP);

int LPROIP_Process(LPDR_HANDLE hROIP, float *pfFeat4D, int adwFeatSZ[4], float *pfRect3D, int adwRectSZ[3]);

int LPROIP_Release(LPDR_HANDLE hROIP);


#endif



