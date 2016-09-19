/*
 * processor_helper.h
 *
 *  Created on: Apr 29, 2016
 *      Author: jiajiachen
 */

#ifndef SRC_PROCESSOR_PROCESSOR_HELPER_H_
#define SRC_PROCESSOR_PROCESSOR_HELPER_H_
#include "processor.h"
#include "watch_dog.h"
#include "c_api.h"
#include "alg/plate/LPDetectRecog.hpp"
#include "fs_util.h"
namespace dg {

static bool roiFilter(const vector<Rect> &mask, Rect src) {
    if (mask.size() == 0)
        return true;

    for (int i = 0; i < mask.size(); i++) {
        Rect tmp = src & mask[i];
        if (tmp.width * tmp.height > 0.5 * src.width * src.height) {
            return true;
        } else {
            continue;
        }

    }
    return false;
}

static int vote(vector<vector<Prediction> > &src, vector<vector<Prediction> > &dst,
                int factor) {
    if (src.size() > dst.size()) {
        for (int i = 0; i < src.size(); i++) {
            vector<Prediction> tmpSrc = src[i];
            vector<Prediction> tmpDst;
            for (int j = 0; j < tmpSrc.size(); j++) {
                tmpDst.push_back(
                    pair<int, float>(tmpSrc[j].first,
                                     tmpSrc[j].second / factor));
            }
            dst.push_back(tmpDst);
        }
        return 1;
    }
    for (int i = 0; i < src.size(); i++) {

        vector<Prediction> tmpSrc = src[i];
        vector<Prediction> tmpDst = dst[i];
        if (tmpSrc.size() != tmpDst.size()) {
            return -1;
        }
        for (int j = 0; j < tmpSrc.size(); j++) {
            tmpDst[j].second += tmpSrc[j].second / factor;
        }
        dst[i] = tmpDst;
    }
    return 1;
}
static bool RecordPerformance(Feature feature, unsigned long long &performance) {
    FILE *fp = NULL;
    if ((fp = fopen(".file_lock.test", "w+")) == NULL)
        DLOG(WARNING) << ("file lock failed!\n");
    int i = flock(fileno(fp), LOCK_EX);
    bool status = false;

    if (i == 0) {
        if (SetCurrPerformance(feature, performance) == ERR_SUCCESS) {
            LOG(INFO) << "write performance11 " << feature << " " << performance << " into dog";

            performance = 0;
            status = true;
        } else {
            LOG(WARNING) << "record feature performance failed" << endl;
            status = false;

        }
        flock(fileno(fp), LOCK_UN);
    }

    fclose(fp);
    return status;

}

static int readTxtFile(const char *pbyFN, char *pbyBuffer, int *pdwBufflen) {
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

static int readBinFileAuto(const char *pbyFN, char **ppbyBuffer, int *pdwBufflen, bool is_encrypt_enabled) {
    char *pbyBuffer = 0;
    char byCh;
    int dwNowLen = 0;

    FILE *pf = fopen(pbyFN, "rb");

    while (!feof(pf)) {

        fread(&byCh, 1, 1, pf);
        dwNowLen++;

    };

    *pdwBufflen = dwNowLen;
    pbyBuffer = (char *) calloc(dwNowLen, 1);
    unsigned char *data = (unsigned char *) calloc(dwNowLen, 1);
    *ppbyBuffer = pbyBuffer;

    dwNowLen = 0;
    fseek(pf, 0, SEEK_SET);
    while (!feof(pf)) {
        fread(&byCh, 1, 1, pf);
        data[dwNowLen] = byCh;
        dwNowLen++;
    };

    fclose(pf);
    if (is_encrypt_enabled) {
        DecryptModel(data, dwNowLen, (unsigned char *) pbyBuffer);
    } else {
        memcpy(pbyBuffer, data, dwNowLen);
    }
    free(data);

    return 0;
}
static Mat CutImage(const Mat &src, Box &box) {
    Mat dst(box.height, box.width, CV_8UC3);
    for (int i = 0; i < box.height; i++) {
        memcpy(dst.data + 3 * (i * box.width),
               src.data + 3 * (box.x + src.cols * i + src.cols * box.y),
               sizeof(uchar) * 3 * box.width);
    }

    return dst;
}
static int readTxtFileAuto(const char *pbyFN, char **ppbyBuffer, int *pdwBufflen, bool is_encrypt_enabled) {
    char *pbyBuffer = 0;
    int dwNowLen = 0;
    dwNowLen = FileSize(pbyFN);
    FILE *fp = fopen(pbyFN, "r");
    unsigned char *buffer = (unsigned char *) calloc(dwNowLen, 1);
    size_t rds = fread(buffer, dwNowLen, 1, fp);
    fclose(fp);
    pbyBuffer = (char *) calloc(dwNowLen, 1);
    if (is_encrypt_enabled) {
        DecryptModel(buffer, dwNowLen, (unsigned char *) pbyBuffer);
    } else {
        memcpy(pbyBuffer, buffer, dwNowLen);
    }
    *ppbyBuffer = pbyBuffer;
    *pdwBufflen = dwNowLen;

    return 0;
}

static int readModuleFile(string symbol_file, string param_file, LPDRModel_S *pstModel, bool is_encrypt_enabled) {
    int dwSymLenDetect = 0;

    readTxtFileAuto(symbol_file.c_str(), &pstModel->pbySym, &dwSymLenDetect, is_encrypt_enabled);

    readBinFileAuto(param_file.c_str(), &pstModel->pbyParam, &pstModel->dwParamSize, is_encrypt_enabled);

    return 0;
}


static int readBinFile(const char *pbyFN, char *pbyBuffer, int *pdwBufflen, bool is_encrypt_enabled) {
    int dwBufferMax = *pdwBufflen;
    char byCh;
    int dwNowLen = 0;

    FILE *pf = fopen(pbyFN, "rb");

    unsigned char *data = (unsigned char *) calloc(dwNowLen, 1);
    while (!feof(pf)) {
        if (dwNowLen > dwBufferMax) {
            printf("no enough buffer!\n");
            break;
        }
        fread(&byCh, 1, 1, pf);
        data[dwNowLen] = byCh;
        dwNowLen++;
    };
    if (is_encrypt_enabled) {
        DecryptModel(data, dwNowLen, (unsigned char *) pbyBuffer);
    } else {
        memcpy(pbyBuffer, data, dwNowLen);
    }
    free(data);

    fclose(pf);

    *pdwBufflen = dwNowLen;

    return 0;
}


static int readFile(const char *pbyFN, mx_float *pfBuffer, mx_uint bufflen) {
    FILE *pf = fopen(pbyFN, "rb");
    fread(pfBuffer, sizeof(mx_float), bufflen, pf);
    fclose(pf);

    return 0;
}


static int ipl2bin(IplImage *pimg, mx_float *pfBuffer, mx_uint bufflen) {
    int width = pimg->width;
    int height = pimg->height;
    int wstep = pimg->widthStep;
    int imgsz = width * height;
    unsigned char *data = (unsigned char *) pimg->imageData;

    assert(imgsz <= bufflen);

    for (int ri = 0; ri < height; ri++) {
        for (int ci = 0; ci < width; ci++) {
            pfBuffer[ri * width + ci] = data[ri * wstep + ci] / 255.f;
        }
    }

    return 0;
}


static void showUBY_IMG(const char *pbyWinName, uchar *pubyImg, int dwImgW, int dwImgH) {
    int dwRI, dwCI;
    IplImage *pcvImg;

    pcvImg = cvCreateImage(cvSize(dwImgW, dwImgH), 8, 1);

    for (dwRI = 0; dwRI < dwImgH; dwRI++) {
        for (dwCI = 0; dwCI < dwImgW; dwCI++) {
            pcvImg->imageData[pcvImg->widthStep * dwRI + dwCI] = pubyImg[dwRI * dwImgW + dwCI];
        }
    }

    //    cvSaveImage("/Users/mzhang/work/LPCE_ERROR/tmp/name.bmp", pcvImg);
    //  cvShowImage(pbyWinName, pcvImg);
    cvReleaseImage(&pcvImg);
}

}

#endif /* SRC_PROCESSOR_PROCESSOR_HELPER_H_ */
