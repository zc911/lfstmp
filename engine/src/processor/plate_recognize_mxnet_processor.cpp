/*
 * plate_recognize_mxnet_processor.cpp
 *
 *  Created on: May 18, 2016
 *      Author: jiajaichen
 */

//#include <matrix_engine/model/model.h>
#include "plate_recognize_mxnet_processor.h"
#include "processor_helper.h"
namespace dg {
/*const char *paInv_chardict[LPDR_CLASS_NUM] = { "_", "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J",
        "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
        "Z", "I", "\u4eac", "\u6d25", "\u6caa", "\u6e1d", "\u5180", "\u8c6b", "\u4e91", "\u8fdb", "\u9ed1", "\u6e58", "皖", "闽",
        "鲁", "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋", "蒙", "\u9655", "吉", "贵", "粤",
        "青", "藏", "川", "宁", "琼", "使", "领", "试", "学", "临", "时", "警", "港", "O",
        "挂", "澳", "#" };*/
const char *paInv_chardict[LPDR_CLASS_NUM] = {"_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
          "A", "B", "C", "D", "E", "F", "G", "H", "J", \
          "K", "L", "M", "N", "P", "Q", "R", "S", "T",\
          "U", "V", "W", "X", "Y", "Z", "I", "京", "津",\
          "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘", \
          "皖", "闽", "鲁", "新", "苏", "浙", "赣", "鄂", \
          "桂", "甘", "晋", "蒙", "陕", "吉", "贵", "粤", \
          "青", "藏", "川", "宁", "琼", "使", "领", "试", \
          "学", "临", "时", "警", "港", "O", "挂", "澳", "#"};
PlateRecognizeMxnetProcessor::PlateRecognizeMxnetProcessor(
        LPDRConfig_S *stConfig)
        : h_LPDR_Handle_(0) {
        setConfig(stConfig);
    LPDR_Create(&h_LPDR_Handle_, stConfig);
}

PlateRecognizeMxnetProcessor::~PlateRecognizeMxnetProcessor() {
    // TODO Auto-generated destructor stub
}
bool PlateRecognizeMxnetProcessor::process(Frame *frame) {
    return false;
}
bool PlateRecognizeMxnetProcessor::process(FrameBatch *frameBatch) {
    float costtime, diff;
#if DEBUG
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif
    int batchsize = batch_size_;
    int imagesize = images_.size();
    for (int i = 0; i < (ceil((float)imagesize / (float)batchsize) * batchsize); i +=
            batchsize) {
        stImgSet_.dwImageNum=0;
        for (int j = 0; j < batchsize; j++) {
            if (i  + j < imagesize) {
                Mat pubyData = images_[i  + j];
                stImgSet_.astSet[j].pubyData = pubyData.data;
                stImgSet_.astSet[j].dwImgH = pubyData.rows;
                stImgSet_.astSet[j].dwImgW = pubyData.cols;
                stImgSet_.dwImageNum++;

            }
        }
        LPDR_OutputSet_S stOutput;
        LPDR_Process(h_LPDR_Handle_, &stImgSet_, &stOutput);
        for (int j = 0; j < batchsize; j++) {
            LPDR_Output_S *pstOut = stOutput.astLPSet+j;
            for (int dwJ = 0; dwJ < pstOut->dwLPNum; dwJ++) {
                LPDRInfo_S *pstLP = pstOut->astLPs + dwJ;
                int *pdwLPRect = pstLP->adwLPRect;
                Vehicle::Plate plate;
                plate.box.x = pdwLPRect[0];
                plate.box.y = pdwLPRect[1];
                plate.box.width = pdwLPRect[2] - pdwLPRect[0];
                   plate.box.height = pdwLPRect[3] - pdwLPRect[1];

                string platenum;
                for (int dwK = 0; dwK < pstLP->dwLPLen; dwK++) {
                    platenum += paInv_chardict[pstLP->adwLPNumber[dwK]];
                }
                plate.color_id = pstLP->dwColor;
                plate.plate_type=pstLP->dwType;
                plate.confidence = pstLP->fAllScore;
                plate.plate_num = platenum;

                Vehicle *v = (Vehicle*) objs_[i + j];
                v->set_plate(plate);
            }

        }

    }
#if DEBUG
    gettimeofday(&end, NULL);
    diff = ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec)
            / 1000.f;
    printf("plate mxnet cost: %.2fms\n", diff);
#endif
    return true;
}

bool PlateRecognizeMxnetProcessor::RecordFeaturePerformance() {
    return true;
}

bool PlateRecognizeMxnetProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if RELEASE
    if(performance_>20000) {
        if(!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    this->vehiclesFilter(frameBatch);
    return true;
}
void PlateRecognizeMxnetProcessor::setConfig(LPDRConfig_S *pstConfig) {
    readModuleFile(pstConfig->fcnnSymbolFile, pstConfig->fcnnParamFile,
                   &pstConfig->stFCNN,pstConfig->is_model_encrypt);
    pstConfig->stFCNN.adwShape[0] = pstConfig->batchsize;
    pstConfig->stFCNN.adwShape[1] = 1;
    pstConfig->stFCNN.adwShape[2] = 400;  //standard width
    pstConfig->stFCNN.adwShape[3] = 400;  //standard height

    readModuleFile(pstConfig->rpnSymbolFile, pstConfig->rpnParamFile,
                   &pstConfig->stRPN,pstConfig->is_model_encrypt);
    pstConfig->stRPN.adwShape[0] = pstConfig->stFCNN.adwShape[0];
    pstConfig->stRPN.adwShape[1] = 4;//number of plates per car;
    pstConfig->stRPN.adwShape[2] = 1;
    pstConfig->stRPN.adwShape[3] = 100;
    pstConfig->stRPN.adwShape[4] = 300;

    readModuleFile(pstConfig->roipSymbolFile, pstConfig->roipParamFile,
                   &pstConfig->stROIP,pstConfig->is_model_encrypt);
    pstConfig->stROIP.adwShape[0] = pstConfig->stRPN.adwShape[0]
            * pstConfig->stRPN.adwShape[1];
    pstConfig->stROIP.adwShape[1] = 0;
    pstConfig->stROIP.adwShape[2] = 0;
    pstConfig->stROIP.adwShape[3] = 0;
    pstConfig->stROIP.adwShape[4] = pstConfig->stROIP.adwShape[0];
    pstConfig->stROIP.adwShape[5] = 20;
    pstConfig->stROIP.adwShape[6] = 5;

    readModuleFile(pstConfig->pregSymbolFile, pstConfig->pregParamFile,
                   &pstConfig->stPREG,pstConfig->is_model_encrypt);
    pstConfig->stPREG.adwShape[0] = 1;
    pstConfig->stPREG.adwShape[1] = 1;
    pstConfig->stPREG.adwShape[2] = 64;
    pstConfig->stPREG.adwShape[3] = 64 * 2;

    readModuleFile(pstConfig->chrecogSymbolFile, pstConfig->chrecogParamFile,
                   &pstConfig->stCHRECOG,pstConfig->is_model_encrypt);
    pstConfig->stCHRECOG.adwShape[0] = 50;
    pstConfig->stCHRECOG.adwShape[1] = 1;
    pstConfig->stCHRECOG.adwShape[2] = 32;
    pstConfig->stCHRECOG.adwShape[3] = 32;

    pstConfig->dwDevType = 2;
    pstConfig->dwDevID = 0;
    batch_size_ = pstConfig->batchsize;
}
void PlateRecognizeMxnetProcessor::vehiclesFilter(FrameBatch *frameBatch) {
    /*   images_.clear();
     images_.push_back(frameBatch->frames()[0]->payload()->data());
     return;*/
    images_.clear();
    objs_.clear();
    objs_ = frameBatch->CollectObjects(FEATURE_CAR_PLATE);
    vector<Object*>::iterator itr = objs_.begin();
    while (itr != objs_.end()) {

        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {
            Vehicle *v = (Vehicle*) obj;
            DLOG(INFO)<< "Put vehicle images to be color classified: " << obj->id() << endl;
            images_.push_back(v->image());
            ++itr;
        } else {
            itr = objs_.erase(itr);
            DLOG(INFO)<<"This is not a type of vehicle: " << obj->id() << endl;
        }
    }
}
}
/* namespace dg */
