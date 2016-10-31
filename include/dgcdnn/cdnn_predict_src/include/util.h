
#ifndef UTIL_H_
#define	UTIL_H_

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <string>
#include "matrix.h"

using namespace std;

typedef std::vector<Matrix*> MatrixV;
typedef std::map<string, Matrix*> MatrixM;
typedef std::vector<float> floatv;
typedef std::vector<int> intv;

typedef vector<map<string, vector<void *> > > listDictParam_t;
typedef map<string, vector<void *> > dictParam_t;

template<typename T>
std::string tostr(T n) {
    std::ostringstream result;
    result << n;
    return result.str();
}

int loadParam(const char *filePath, listDictParam_t &listDictParam, int isMultiPatch = 0);

int releaseParam();


floatv* getFloatV(vector<void *> &vecSrc);

intv* getIntV(vector<void *> &vecSrc);

MatrixV* getMatrixV(vector<void *> &vecSrc);

int dictGetInt(dictParam_t &dict, const char* key);

intv* dictGetIntV(dictParam_t &dict, const char* key);

string dictGetString(dictParam_t &dict, const char* key);

float dictGetFloat(dictParam_t &dict, const char* key);

floatv* dictGetFloatV(dictParam_t &dict, const char* key);

Matrix* dictGetMatrix(dictParam_t &dict, const char* key);

MatrixV* dictGetMatrixV(dictParam_t &dict, const char* key);



#endif	/* UTIL_H_ */



