#include <iostream>
#include <fstream>
#include <string>
#include "LPDetectRecog.hpp"

#include "cv.h"
#include "highgui.h"

#define BSHOW 0

#define LPDR_CLASS_NUM 79

#define TEST_NUM 1

#define THREAD_NUM 40

#if 1 //dog

#if 1 //full
#define BATCH_SIZE 4
#define STD_IMG_H 1080 //2600
#define STD_IMG_W 1920 //2600
#define ROI_PER_IMG 16
#define RPN_WIDTH 300
#define RPN_HEIGHT 100
#else //car
#define BATCH_SIZE 1
#define STD_IMG_H 600
#define STD_IMG_W 400
#define ROI_PER_IMG 4
#define RPN_WIDTH 300
#define RPN_HEIGHT 100
#endif

//#define LPDR_MDL "MDL/0725"
#define LPDR_MDL "MDL/0728_2"

#else //dog

#if 1 //full
#define BATCH_SIZE 1
#define STD_IMG_H 600 //2600
#define STD_IMG_W 1200 //2600
#define ROI_PER_IMG 16
#define RPN_WIDTH 300
#define RPN_HEIGHT 100
#else //car
#define BATCH_SIZE 16
#define STD_IMG_H 400//600
#define STD_IMG_W 300//400
#define ROI_PER_IMG 2
#define RPN_WIDTH 300
#define RPN_HEIGHT 100
#endif

#define LPDR_MDL "MDL/0728"

#endif //dog


const char *paColors[9] = {"WHITE", "SILVER", "YELLOW", "PINK", "RED", "GREEN", "BLUE", "BROWN", "BLACK"};

const char *paInv_chardict2[LPDR_CLASS_NUM] = {"_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
          "A", "B", "C", "D", "E", "F", "G", "H", "J", \
          "K", "L", "M", "N", "P", "Q", "R", "S", "T",\
          "U", "V", "W", "X", "Y", "Z", "I", "京", "津",\
          "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘", \
          "皖", "闽", "鲁", "新", "苏", "浙", "赣", "鄂", \
          "桂", "甘", "晋", "蒙", "陕", "吉", "贵", "粤", \
          "青", "藏", "川", "宁", "琼", "使", "领", "试", \
          "学", "临", "时", "警", "港", "O", "挂", "澳", "#"};

void showUBY_IMG(const char *pbyWinName, uchar *pubyImg, int dwImgW, int dwImgH);
int readFile(const char *pbyFN, mx_float *pfBuffer, mx_uint bufflen);
int ipl2bin(IplImage *pimg, mx_float *pfBuffer, mx_uint bufflen);

int readTxtFile(const char *pbyFN, char *pbyBuffer, int *pdwBufflen);
int readBinFile(const char *pbyFN, char *pbyBuffer, int *pdwBufflen);
int read_show(string &fn);
int test_imglist(string &fname, string errfile, string resultfile);

int readModuleFile(string symbol_file, string param_file, LPDRModel_S *pstModel);
int setConfig(LPDRConfig_S *pstConfig);

int doRotate_8UC3(uchar *pubyImage, int dwW, int dwH, float fAngle);

//注意：当字符串为空时，也会返回一个空字符串  
void split(string& s, string& delim, vector<string> *ret);

int threadstest();

int main(int argc, char* argv[]) {

#if 0
//	string fname(argv[1]);
//  string fname = "/home/mingzhang/Pictures/test.list";
//  string fname = "/home/mingzhang/DeepV_Data.LPR_Data.TestData.DoubleRowPlate.list";
//  string fname = "/media/mingzhang/3663c8d5-bd82-4820-a2cc-2d4411798d03/mingzhang/work/data/deepv_for_mingzhang/data1/carplatemark1.txt";
//  string fname = "/home/mingzhang/work/LPRProject/DetectRecog/carplatemark1_huang.txt";
//  string fname = "/home/mingzhang/work/LPRProject/DetectRecog/cars_huang.txt";
//  string fname = "err.list";
//  string fname = "cartest.list";
  string fname = "fulltest.list";
//  string fname = "/media/mingzhang/3663c8d5-bd82-4820-a2cc-2d4411798d03/mingzhang/work/data/shoufeizhanpart.list";
//  string fname = "/home/mingzhang/Pictures/xjc2.list";
//  string fname = "/home/mingzhang/Pictures/test_xj.list";

	cout << fname << endl;
	
	string errfile = "";//"err.list";
	string resultfile = "";
	
	test_imglist(fname, errfile, resultfile);
#else
  threadstest(); 
#endif

  cout << "bye bye ..." << endl;

	return 0;
}


void *threadone(void*pParam);
int threadstest()
{
  pthread_t tids[THREAD_NUM];
  for (int i = 0; i < THREAD_NUM; i++)
  {
    printf("create thread:%d\n", i);
    pthread_create(&tids[i], NULL, threadone, NULL);
  }
  for (int i = 0; i < THREAD_NUM; i++)
  {
    pthread_join(tids[i], NULL);
  }
}


void *threadone(void*pParam)
{
//  string fname = "/home/mingzhang/work/LPRProject/DetectRecog/cars_huang.txt";
//  string fname = "/media/mingzhang/3663c8d5-bd82-4820-a2cc-2d4411798d03/mingzhang/work/data/shoufeizhanall.list";
//  string fname = "/home/mingzhang/DeepV_Data.Brand_Data.company.ZKTD.Passing.20160313.list";
  string fname = "/media/mingzhang/3663c8d5-bd82-4820-a2cc-2d4411798d03/mingzhang/work/data/ZKTD/Passing/20160313.list";
	string errfile = "";//"err.list";
	string resultfile = "";
	
	test_imglist(fname, errfile, resultfile);
}


int test_imglist(string &fname, string errfile, string resultfile) {
	FILE *pf = fopen(fname.c_str(), "r");
	
	if (!pf) {
		printf("cannot open file.\n");
		return -1;
	}

	LPDR_OutputSet_S stOutput;
	LPDR_HANDLE hLPDRHandle = 0;

	LPDRConfig_S stConfig;

	setConfig(&stConfig);

	float fZoom = 1.0f;
	int zeronum = 0;

	CvFont font1;
	cvInitFont(&font1, CV_FONT_HERSHEY_SIMPLEX, 0.47, 0.47, 0, 1, 8);
  
  LPDR_Create(&hLPDRHandle, &stConfig);
//  cout << "fuck 1\n";

  float costtime, diff;
  struct timeval start2, end2, start, end;
  
  ifstream infile;
  ofstream outfile, outresult;
  
  string strfilename;
  string previousLine="";
	char filename[256];
	char abyText[256];
	string strSplit = " ";
	int num = 0;
	int errnum = 0;
	int realnum = 0;
	int dwImgH = 0, dwImgW = 0;
	int dwImgH2 = 0, dwImgW2 = 0;

//	cv::namedWindow("hello_rgb", cv::WINDOW_AUTOSIZE);
  if (BSHOW)
  {
	  cv::namedWindow("hello_rgb", cv::WINDOW_NORMAL);
  }
//	fscanf(pf, "%s", filename);
	cv::Mat imgcolor;
	uchar *pubyImgData = 0;
	vector<string> strparts;

	LPDR_ImageSet_S stImgSet;

  stImgSet.dwImageNum = stConfig.stFCNN.adwShape[0];
  
  infile.open(fname);
  
  if (errfile != "")
  {
    outfile.open(errfile);
  }
  
  if (resultfile != "")
  {
    outresult.open(resultfile);
  }
  
  int dwAllLPCount = 0;
  int dwRightLPCount = 0;
  int dwRecallCount = 0;
//	while (fscanf(pf, "%s", filename) != EOF) {
	while (1) {
	  getline(infile, strfilename);
	  if (infile.eof()) break;
		cout << strfilename << endl;
		split(strfilename, strSplit, &strparts);
		
//		for (int si = 0; si < strparts.size(); si++)
//		{
//		  cout << strparts[si] << endl;
//		}
	  cv::Mat imgcolor_ori = cv::imread(strparts[0], CV_LOAD_IMAGE_COLOR);
    if (imgcolor_ori.rows == 0) {
      printf("no more pictures ...\n");
      continue;
    }
    
		num++;
		printf("frame: %d\n", num);
		
		dwAllLPCount += strparts.size() - 1;
		
		if (1) {
			dwImgH2 = imgcolor_ori.rows;
			dwImgW2 = imgcolor_ori.cols;

			cv::resize(imgcolor_ori, imgcolor, cv::Size(dwImgW2*fZoom, dwImgH2*fZoom), 0, 0, CV_INTER_LINEAR);
			dwImgH = imgcolor.rows;
			dwImgW = imgcolor.cols;

      if (0)
      {
        cv::Mat borderImg(dwImgH+20, dwImgW+20, CV_8UC3);
        cv::copyMakeBorder(imgcolor, borderImg, 0, 20, 0, 20, cv::BORDER_CONSTANT, 0);
        cv::imshow("fuck", borderImg);
        cv::waitKey(0);
      }
		}
    
		uchar *pubyData = (uchar*)imgcolor.data;
    
    for (int dwBI = 0; dwBI < stConfig.stFCNN.adwShape[0]; dwBI++)
    {
		  stImgSet.astSet[dwBI].pubyData = pubyData;
		  stImgSet.astSet[dwBI].dwImgW = dwImgW;
		  stImgSet.astSet[dwBI].dwImgH = dwImgH;
		}
    
		gettimeofday(&start, NULL);
		for (int kk = 0; kk < TEST_NUM; kk++)
		{
		  gettimeofday(&start2, NULL);
		  LPDR_Process(hLPDRHandle, &stImgSet, &stOutput);
		  gettimeofday(&end2, NULL);
		  diff = ((end2.tv_sec-start2.tv_sec)*1000000+ end2.tv_usec-start2.tv_usec) / 1000.f;
		  cout << kk << ":" << diff << endl;
		}
		gettimeofday(&end, NULL);
		diff = ((end.tv_sec-start.tv_sec)*1000000+ end.tv_usec-start.tv_usec) / 1000.f;
		printf("LPDR_Process cost:%.2fms\n", diff/TEST_NUM);
		
		char key;
		int dwMargin;
//		cv::Mat images[3] = {imgcolor, rszImg, rszImg2};
    cv::Mat images[8] = {imgcolor, imgcolor, imgcolor, imgcolor, imgcolor, imgcolor, imgcolor, imgcolor};
    
    int dwNowRightCount = 0;
    
		for (int dwI = 0; dwI < stOutput.dwImageNum; dwI++)
		{
		  cout << "image:" << dwI << endl;
		  LPDR_Output_S *pstOut = stOutput.astLPSet + dwI;
		  cv::Mat &imgnow = images[dwI];
		  string strallinfo = "";
		  for (int dwJ = 0; dwJ < pstOut->dwLPNum; dwJ++)
		  {
		    LPDRInfo_S *pstLP = pstOut->astLPs + dwJ;
		    int *pdwLPRect = pstLP->adwLPRect;
		    dwMargin = (pdwLPRect[3] - pdwLPRect[1])/4;
		    int dwX0 = pdwLPRect[0] - dwMargin;
		    int dwY0 = pdwLPRect[1] - dwMargin;
		    int dwX1 = pdwLPRect[2] + dwMargin;
		    int dwY1 = pdwLPRect[3] + dwMargin;
        
        if (BSHOW)
        {
		      cv::rectangle(imgnow, cv::Point(dwX0, dwY0), cv::Point(dwX1, dwY1), CV_RGB(255, 0, 0), 2, 8, 0);
        }
        string text = "";
		    for (int dwK = 0; dwK < pstLP->dwLPLen; dwK++)
		    {
		      cout << paInv_chardict2[pstLP->adwLPNumber[dwK]];
          text += paInv_chardict2[pstLP->adwLPNumber[dwK]];
		    }

        cout << endl;

        for (int dwK = 0; dwK < pstLP->dwLPLen; dwK++)
		    {
		      cout << setprecision(6) << pstLP->afScores[dwK] << ",";
		    }

        cout << endl;

		    stringstream sspos;
		    string strpos;
		    sspos << dwX0 << "," << dwY0 << "," << dwX1 << "," << dwY1 << ";";
		    sspos >> strpos;
		    strallinfo += ";" + strpos + text;
        
		    if (BSHOW)
        {
          CvFont fontB = cv::fontQt("Times", 20, cv::Scalar(255, 0, 0));
          cv::rectangle(imgnow, cv::Point(dwX0, dwY0-25), cv::Point(dwX0+120, dwY0), CV_RGB(255, 255, 255), CV_FILLED);
          cv::addText(imgnow, text, cv::Point(dwX0, dwY0-4), fontB);
        }

		    bool bRight = false;
		    for (int mi = 1; mi < strparts.size(); mi++)
		    {
		      string platenow = strparts[mi];
		      string recogplate = "";
		      for (int dwK = 0; dwK < pstLP->dwLPLen; dwK++)
		      {
		        recogplate += paInv_chardict2[pstLP->adwLPNumber[dwK]];
		      }
           
          
		      int dwCmp = platenow.compare(recogplate);
//		      cout << "compare:" << platenow << ", " << recogplate << ": " <<  << endl;
          if (dwCmp == 0)
          {
            bRight = true;
            break;
          }
		    }
		    
		    cout << " " << pstLP->fAllScore << ", type:" << pstLP->dwType << ", color:" << paColors[pstLP->dwColor];
		    if (bRight)
        {
          dwRightLPCount++;
          dwNowRightCount++;
          cout << "--------Right!";
        }
        cout << endl;
		  }
		  
		  if (resultfile != "")
		  {
		    outresult << strfilename << strallinfo << endl;
		  }
		  
		  if (errfile != "")
		  {
		    if (dwNowRightCount != strparts.size() - 1)
		    {
		      outfile << strfilename << endl;
		    }
		  }
		  printf("========>>Right_Ratio: %.2f%%[%d/%d]\n", dwRightLPCount*100.0f/dwAllLPCount, dwRightLPCount, dwAllLPCount);
      if (BSHOW)
      {
		    cv::imshow("hello_rgb", imgnow);
		    key = cv::waitKey(0);
      }
		}

		if (key == 'c' || key == 'C') {
			break;
		}
		else if (key == ' ') {
		  cv::waitKey(0);
		}
//		cout << "fuck test 3\n";
	}
  
//  cout << "fuck test end\n";
  infile.close();
  
  if (errfile != "")
  {
    outfile.close();
  }

  if (resultfile != "")
  {
    outresult.close();
  }

	LPDR_Release(hLPDRHandle);


	return 0;
}


int setConfig(LPDRConfig_S *pstConfig)
{
//  string pfolder = "./MDL/0510";
//  string pfolder = "./MDL/0524";
//  string pfolder = "./MDL/0630";
//  string pfolder = "./MDL/0630_2";
//  string pfolder = "./MDL/0701";
//  string pfolder = "./MDL/0713";
//  string pfolder = "./MDL/0714";
//  string pfolder = "./MDL/0720";
  string pfolder = LPDR_MDL;//"./MDL/0725";
    
  string fcn_symbol_file = pfolder + "/fcn.symbol";
	string fcn_param_file = pfolder + "/fcn_params.bin";
	readModuleFile(fcn_symbol_file, fcn_param_file, &pstConfig->stFCNN);
	pstConfig->stFCNN.adwShape[0] = BATCH_SIZE;//6;
	pstConfig->stFCNN.adwShape[1] = 1;
	pstConfig->stFCNN.adwShape[2] = STD_IMG_H;//2600;
	pstConfig->stFCNN.adwShape[3] = STD_IMG_W;//2600;
	
	string rpn_symbol_file = pfolder + "/rpn.symbol";
	string rpn_param_file = pfolder + "/rpn_params.bin";
	readModuleFile(rpn_symbol_file, rpn_param_file, &pstConfig->stRPN);
	pstConfig->stRPN.adwShape[0] = pstConfig->stFCNN.adwShape[0];
	pstConfig->stRPN.adwShape[1] = ROI_PER_IMG;//16;
	pstConfig->stRPN.adwShape[2] = 1;
	pstConfig->stRPN.adwShape[3] = RPN_HEIGHT;//120;
	pstConfig->stRPN.adwShape[4] = RPN_WIDTH;//360;
	
	string roip_symbol_file = pfolder + "/roip.symbol";
	string roip_param_file = pfolder + "/roip_params.bin";
	readModuleFile(roip_symbol_file, roip_param_file, &pstConfig->stROIP);
	pstConfig->stROIP.adwShape[0] = pstConfig->stRPN.adwShape[0] * pstConfig->stRPN.adwShape[1];
	pstConfig->stROIP.adwShape[1] = 0;
	pstConfig->stROIP.adwShape[2] = 0;
	pstConfig->stROIP.adwShape[3] = 0;
	pstConfig->stROIP.adwShape[4] = pstConfig->stROIP.adwShape[0];
	pstConfig->stROIP.adwShape[5] = 20;
	pstConfig->stROIP.adwShape[6] = 5;
	
	
	string polyreg_symbol_file = pfolder + "/polyreg.symbol";
	string polyreg_param_file = pfolder + "/polyreg_params.bin";
	readModuleFile(polyreg_symbol_file, polyreg_param_file, &pstConfig->stPREG);
	pstConfig->stPREG.adwShape[0] = 1;
	pstConfig->stPREG.adwShape[1] = 1;
	pstConfig->stPREG.adwShape[2] = 64;
	pstConfig->stPREG.adwShape[3] = 64*2;
	
	string chrecog_symbol_file = pfolder + "/chrecog.symbol";
	string chrecog_param_file = pfolder + "/chrecog_params.bin";
	readModuleFile(chrecog_symbol_file, chrecog_param_file, &pstConfig->stCHRECOG);
	pstConfig->stCHRECOG.adwShape[0] = 50;
	pstConfig->stCHRECOG.adwShape[1] = 1;
	pstConfig->stCHRECOG.adwShape[2] = 32;
	pstConfig->stCHRECOG.adwShape[3] = 32;
	
	pstConfig->dwDevType = 2;
	pstConfig->dwDevID = 0;
}


int readTxtFileAuto(const char *pbyFN, char **ppbyBuffer, int *pdwBufflen);
int readBinFileAuto(const char *pbyFN, char **ppbyBuffer, int *pdwBufflen);

int readModuleFile(string symbol_file, string param_file, LPDRModel_S *pstModel)
{
  int dwSymLenDetect = 0;
    
	readTxtFileAuto(symbol_file.c_str(), &pstModel->pbySym, &dwSymLenDetect);
	
	readBinFileAuto(param_file.c_str(), &pstModel->pbyParam, &pstModel->dwParamSize);

    return 0;
}


int readTxtFile(const char *pbyFN, char *pbyBuffer, int *pdwBufflen) {
  int dwBufferMax = *pdwBufflen;
  char byCh;
  int dwNowLen = 0;

  FILE *pf = fopen(pbyFN, "r");

  while (!feof(pf)) {
    if (dwNowLen > dwBufferMax) {
  		printf("no enough buffer!\n");
  		break;
  	}
  	byCh = fgetc(pf);
  	pbyBuffer[dwNowLen] = byCh;
  	dwNowLen++;
  };

  fclose(pf);
  
  *pdwBufflen = dwNowLen;
  
  return 0;
}



int readTxtFileAuto(const char *pbyFN, char **ppbyBuffer, int *pdwBufflen) {
  char *pbyBuffer = 0;
  char byCh;
  int dwNowLen = 0;

  FILE *pf = fopen(pbyFN, "r");

  while (!feof(pf)) {
  	byCh = fgetc(pf);
  	dwNowLen++;
  };
  
  *pdwBufflen = dwNowLen;
  pbyBuffer = (char*)calloc(dwNowLen, 1);
  *ppbyBuffer = pbyBuffer;
  
  dwNowLen = 0;
  fseek(pf, 0, SEEK_SET);
  while (!feof(pf)) {
  	byCh = fgetc(pf);
  	pbyBuffer[dwNowLen] = byCh;
  	dwNowLen++;
  };

  fclose(pf);
  
  *pdwBufflen = dwNowLen;
  
  return 0;
}



int readBinFile(const char *pbyFN, char *pbyBuffer, int *pdwBufflen) {
	int dwBufferMax = *pdwBufflen;
  char byCh;
  int dwNowLen = 0;

  FILE *pf = fopen(pbyFN, "rb");

  while (!feof(pf)) {
  	if (dwNowLen > dwBufferMax) {
  		printf("no enough buffer!\n");
  		break;
  	}
  	fread(&byCh, 1, 1, pf);
  	pbyBuffer[dwNowLen] = byCh;
  	dwNowLen++;
  };

  fclose(pf);
  
  *pdwBufflen = dwNowLen;
  
  return 0;
}


int readBinFileAuto(const char *pbyFN, char **ppbyBuffer, int *pdwBufflen) {
  char *pbyBuffer = 0;
  char byCh;
  int dwNowLen = 0;

  FILE *pf = fopen(pbyFN, "rb");
  
  while (!feof(pf)) {
  	fread(&byCh, 1, 1, pf);
  	dwNowLen++;
  };
  
  *pdwBufflen = dwNowLen;
  pbyBuffer = (char*)calloc(dwNowLen, 1);
  *ppbyBuffer = pbyBuffer;
  
  dwNowLen = 0;
  fseek(pf, 0, SEEK_SET);
  while (!feof(pf)) {
  	fread(&byCh, 1, 1, pf);
  	pbyBuffer[dwNowLen] = byCh;
  	dwNowLen++;
  };

  fclose(pf);

  return 0;
}


int readFile(const char *pbyFN, mx_float *pfBuffer, mx_uint bufflen) {
  FILE *pf = fopen(pbyFN, "rb");
  fread(pfBuffer, sizeof(mx_float), bufflen, pf);
  fclose(pf);
  
  return 0;
}


int ipl2bin(IplImage *pimg, mx_float *pfBuffer, mx_uint bufflen) {
	int width = pimg->width;
	int height = pimg->height;
	int wstep = pimg->widthStep;
	int imgsz = width * height;
	unsigned char *data = (unsigned char*)pimg->imageData;
	
	assert(imgsz <= bufflen);
	
	for (int ri = 0; ri < height; ri++) {
		for (int ci = 0; ci < width; ci++) {
			pfBuffer[ri * width + ci] = data[ri * wstep + ci] / 255.f;
		}
	}
	
	return 0;
}


//注意：当字符串为空时，也会返回一个空字符串  
void split(string& s, string& delim, vector<string> *ret)
{  
    size_t last = 0;
    size_t index=s.find_first_of(delim,last);  
    ret->clear();
    while (index!=string::npos)  
    {
        ret->push_back(s.substr(last,index-last));  
        last=index+1;  
        index=s.find_first_of(delim,last);  
    }  
    if (index-last>0)  
    {  
        ret->push_back(s.substr(last,index-last));  
    }  
}


void showUBY_IMG(const char *pbyWinName, uchar *pubyImg, int dwImgW, int dwImgH)
{
    int dwRI, dwCI;
    IplImage *pcvImg;
    
    pcvImg = cvCreateImage(cvSize(dwImgW, dwImgH), 8, 1);
    
    for (dwRI = 0; dwRI < dwImgH; dwRI++) {
        for (dwCI = 0; dwCI < dwImgW; dwCI++) {
            pcvImg->imageData[pcvImg->widthStep * dwRI + dwCI] = pubyImg[dwRI * dwImgW + dwCI];
        }
    }
    
    //    cvSaveImage("/Users/mzhang/work/LPCE_ERROR/tmp/name.bmp", pcvImg);
    cvShowImage(pbyWinName, pcvImg);
    cvReleaseImage(&pcvImg);
}


/*
int doRotate_8UC3(uchar *pubyImage, int dwW, int dwH, float fAngle)
{
  cv::Mat matRotate(2, 3, CV_8UC3);
  cv::Mat matImageSrc(dwH, dwW, CV_8UC3, pubyImage);
  cv::Mat matImageDst(dwH, dwW, CV_8UC3);
  matRotate = getRotationMatrix2D(cv::Point(dwW/2, dwH/2), fAngle, 1.0);
  warpAffine(matImageSrc, matImageDst, matRotate, matImageDst.size());
  matImageDst.copyTo(matImageSrc);
  
  return 0;
}
*/

