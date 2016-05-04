/*
 * plate_recognizer.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "plate_recognizer.h"

namespace dg {

const int WIDTH = 2560;
const int HEIGHT = 2048;

PlateRecognizer::PlateRecognizer(const PlateConfig &config) {
    mem1 = new unsigned char[0x4000];
    mem2 = new unsigned char[40 * 1024 * 1024];
    c_Config = c_defConfig;
    c_Config.nMinPlateWidth = config.MinWidth;
    c_Config.nMaxPlateWidth = config.MaxWidth;
    c_Config.bMovingImage = config.IsMovingImage;
    c_Config.bOutputSingleFrame = 1;
    c_Config.nMaxImageWidth = WIDTH;
    c_Config.nMaxImageHeight = HEIGHT;
    c_Config.pFastMemory = mem1;
    c_Config.nFastMemorySize = 0x4000;
    c_Config.pMemory = mem2;
    c_Config.nMemorySize = 40 * 1024 * 1024;
    c_Config.bIsFieldImage = 0;
    c_Config.bUTF8 = 1;

    int nRet = TH_InitPlateIDSDK(&c_Config);
    LOG(INFO)<<"nREt: "<<nRet<<endl;
    if (nRet != 0) {
        LOG(INFO)<<("nRet = %d, try sudo ./program\n", nRet)<<endl;
        exit(-1);
    }
    nRet = TH_SetProvinceOrder((char *) config.LocalProvince.c_str(),
                               &c_Config);
    nRet = TH_SetRecogThreshold(5, 2, &c_Config);
    LOG(INFO)<<"TH_SetRecogThreshold "<<nRet<<endl;
    nRet = TH_SetImageFormat(ImageFormatBGR, 0, 0, &c_Config);
    LOG(INFO)<<"TH_SetImageFormat "<<nRet<<endl;
    LOG(INFO)<<"TH_SetImageFormat "<<nRet<<endl;

}

PlateRecognizer::~PlateRecognizer() {
    delete mem1;
    delete mem2;
}
int PlateRecognizer::recognizeImage(const Mat &img) {
    Mat sample;

    if (img.channels() == 4)
        cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1)
        cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    if (sample.channels() != 3) {
        LOG(INFO)<<"Sample color error"<<sample.channels()<<endl;
    }

    // TODO resize if image too large
//    if ((sample.rows > 1000) || (sample.cols > 1000)) {
//        return -1;
//    }

    unsigned char *pImg = new unsigned char[sample.rows * sample.cols * 3];

    int cnt = 0;
    for (int i = 0; i < sample.rows; i++) {
        for (int j = 0; j < sample.cols; j++) {
            for (int c = 0; c < sample.channels(); c++) {
                pImg[cnt++] = sample.at<uchar>(i, j * 3 + c);
            }
        }
    }

    int nResultNum = 1;

    int nRet = TH_RecogImage(pImg, sample.cols, sample.rows, &result,
                             &nResultNum, NULL, &c_Config);

    delete[] pImg;

    if (nRet != 0) {
        LOG(WARNING)<<"Plate recognizer error : "<<nRet<<endl;
    }
    return nRet;
}

Vehicle::Plate PlateRecognizer::Recognize(const Mat &img) {
    if (nRet != 0) {
        LOG(INFO)<<"plate recognizer error : "<<nRet<<endl;
    }

    recognizeImage(img);

    Vehicle::Plate plate;
    plate.plate_num = result.license;
    plate.color_id = result.nColor;
    plate.plate_type = result.nType;
    plate.confidence = result.nConfidence / 100.0;

    Box cutboard;
    cutboard.x=result.rcLocation.left;
    cutboard.y=result.rcLocation.top;
    cutboard.width=result.rcLocation.right-result.rcLocation.left;
    cutboard.height=result.rcLocation.bottom-result.rcLocation.top;
    plate.box=cutboard;
    return plate;
}
vector<Vehicle::Plate> PlateRecognizer::RecognizeBatch(
        const vector<Mat> &imgs) {
    if (nRet != 0) {
        LOG(INFO)<<"plate recognizer error"<<endl;
    }
    vector<Vehicle::Plate> vRecognizeResult;

    for(int i=0;i<imgs.size();i++) {
        Mat sample;
        Mat img = imgs.at(i);
        Vehicle::Plate plate;

        if (img.channels() == 4)
        cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1)
        cvtColor(img, sample, CV_GRAY2BGR);
        else
        sample = img;

        if(sample.channels() != 3) {
            LOG(INFO)<<"Sample color error"<<sample.channels()<<endl;
        }

        unsigned char *pImg = new unsigned char[sample.rows * sample.cols * 3];

        int cnt = 0;
        for (int i = 0; i < sample.rows; i++) {
            for (int j = 0; j < sample.cols; j++) {
                for (int c = 0; c < sample.channels(); c++) {
                    pImg[cnt++] = sample.at<uchar>(i, j * 3 + c);
                }
            }
        }

        int width = sample.cols;
        int height = sample.rows;
        int nResultNum = 1;
        int nRet = TH_RecogImage(pImg, width, height, &result, &nResultNum, NULL, &c_Config);

        delete[] pImg;

        if(nRet!=0) {
            LOG(INFO)<<"plate recognizer error No"<<i<<" image: "<<nRet<<endl;
        }

        plate.plate_num=result.license;
        plate.color_id=result.nColor;
        plate.plate_type=result.nType;
        plate.confidence=result.nConfidence;

        Box cutboard;
        cutboard.x=result.rcLocation.left;
        cutboard.y=result.rcLocation.top;
        cutboard.width=result.rcLocation.right-result.rcLocation.left;
        cutboard.height=result.rcLocation.bottom-result.rcLocation.top;
        plate.box=cutboard;
        vRecognizeResult.push_back(plate);
    }
    return vRecognizeResult;
}
void PlateRecognizer::Init(void *config) {
    int nthreads;
    TH_GetKeyMaxThread(&nthreads);

}

} /* namespace dg */
