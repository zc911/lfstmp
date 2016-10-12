// Util.h

#ifndef __UTIL__
#define __UTIL__

#define USING_OPENCV
#include <string>
#include <vector>
#ifdef USING_OPENCV
	#include "highgui.h"
#endif
namespace Cdnn{
namespace Util
{

	#define ERROR_FILE			"my_error.log"
	
	#define FREE(f) if((f)) {free((f)); (f)=NULL;}

	void StartRand();
	double GetRand_64f(double dMin, double dMax);

	//////////// file path operation /////////////

	std::string ReplaceString(std::string srcString, std::string srcSubStr, std::string replaceSubStr);

	std::string GetFileName(std::string filePath);
	// 得到文件路径扩展名外的部分，不包含'.'
	// example：filePath 为 e:\example\example1\example.h
	// 返回为h
	std::string GetFileExt(std::string filePath);
	std::string GetFilePathExceptExt(std::string filePath);
	std::string GetFileDir(std::string filePath);

	///////////// save/load text to file ///////////////////
	bool AppendTxtToFile(const char* outStr, const char* filePath);
	bool SaveData_64f(const char* desPath, const double* pData, int width, int height);

	////////////// display on image ///////////////////////
#ifdef USING_OPENCV
	void ShowPoint(const char* desPath, const double* landmarks, int point);
	void DrawPoint(IplImage*img, double x, double y, CvScalar color);
	void DrawPointArray(IplImage*img, const double* pointsArray, int nPtNum, CvScalar color, bool doesShowOrder);

	void DrawRect(IplImage*img, CvRect box, CvScalar color);
	void DrawRectArray(IplImage*img, const CvRect* rectArray, int nRectNum, CvScalar ptColor);
#endif

	void split(const std::string& src, const std::string& separator, std::vector<std::string>& dest);

}
}
#endif
