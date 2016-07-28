/*
 * plate_recognize_mxnet_processor.cpp
 *
 *  Created on: May 18, 2016
 *      Author: jiajaichen
 */

//#include <matrix_engine/model/model.h>
#include "plate_recognize_mxnet_processor.h"
#include "debug_util.h"

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
                                              "K", "L", "M", "N", "P", "Q", "R", "S", "T", \
                                              "U", "V", "W", "X", "Y", "Z", "I", "京", "津", \
                                              "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘", \
                                              "皖", "闽", "鲁", "新", "苏", "浙", "赣", "鄂", \
                                              "桂", "甘", "晋", "蒙", "陕", "吉", "贵", "粤", \
                                              "青", "藏", "川", "宁", "琼", "使", "领", "试", \
                                              "学", "临", "时", "警", "港", "O", "挂", "澳", "#"
                                             };
PlateRecognizeMxnetProcessor::PlateRecognizeMxnetProcessor(PlateRecognizeMxnetConfig *config
                                                          )
    : config_(config), h_LPDR_Handle_(0) {
    LPDRConfig_S stConfig;
    setConfig(&stConfig);
    LPDR_Create(&h_LPDR_Handle_, &stConfig);
}

PlateRecognizeMxnetProcessor::~PlateRecognizeMxnetProcessor() {
    // TODO Auto-generated destructor stub
    LPDR_Release(h_LPDR_Handle_);
}
bool PlateRecognizeMxnetProcessor::process(Frame *frame) {
    return false;
}
bool PlateRecognizeMxnetProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start Plate Recognize Mxnet Processor : " << frameBatch->id() << endl;

    float costtime, diff;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    int batchsize = batch_size_;
    int imagesize = images_.size();
    VLOG(VLOG_RUNTIME_DEBUG) << "LPDR loop: " << (ceil((float) imagesize / (float) batchsize) * batchsize) << endl;
    for (int i = 0; i < (ceil((float) imagesize / (float) batchsize) * batchsize); i +=
                batchsize) {
        stImgSet_.dwImageNum = 0;
        for (int j = 0; j < batchsize; j++) {
            if (i + j < imagesize) {
                Mat pubyData = images_[i + j];
                stImgSet_.astSet[j].pubyData = pubyData.data;
                stImgSet_.astSet[j].dwImgH = pubyData.rows;
                stImgSet_.astSet[j].dwImgW = pubyData.cols;
                stImgSet_.dwImageNum++;

            }
        }
        LPDR_OutputSet_S stOutput;

        VLOG(VLOG_RUNTIME_DEBUG) << "Start LPDR: " << frameBatch->id() << endl;
        LPDR_Process(h_LPDR_Handle_, &stImgSet_, &stOutput);
        VLOG(VLOG_RUNTIME_DEBUG) << "Finish LPDR: " << frameBatch->id() << endl;


        VLOG(VLOG_RUNTIME_DEBUG) << "Start Post process: " << frameBatch->id() << endl;

        for (int j = 0; j < batchsize; j++) {
            LPDR_Output_S *pstOut = stOutput.astLPSet + j;
            vector<Vehicle::Plate> plates;
            for (int dwJ = 0; dwJ < pstOut->dwLPNum; dwJ++) {
                LPDRInfo_S *pstLP = pstOut->astLPs + dwJ;
                int *pdwLPRect = pstLP->adwLPRect;
                Vehicle::Plate plate;
                plate.box.x = pdwLPRect[0];
                plate.box.y = pdwLPRect[1];
                plate.box.width = pdwLPRect[2] - pdwLPRect[0];
                plate.box.height = pdwLPRect[3] - pdwLPRect[1];

                string platenum;
                float score = 0;
                if (enable_local_province_) {
                    if (pstLP->afScores[0] < local_province_confidence_) {
                        platenum = local_province_;
                    } else {
                        platenum = paInv_chardict[pstLP->adwLPNumber[0]];
                    }
                    score += pstLP->afScores[0];
                }
                for (int dwK = 1; dwK < pstLP->dwLPLen; dwK++) {
                    platenum += paInv_chardict[pstLP->adwLPNumber[dwK]];
                    score += pstLP->afScores[dwK];
                }


                score /= pstLP->dwLPLen;
                plate.color_id = pstLP->dwColor;
                plate.plate_type = pstLP->dwType;
                plate.confidence = score;
                plate.plate_num = platenum;
                plate.local_province_confidence = pstLP->afScores[0];
                plates.push_back(plate);
            }
            if (objs_.size() > (i + j)) {
                Vehicle *v = (Vehicle *) objs_[i + j];
                v->set_plates(plates);
            }

        }

    }
    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Plate mxnet cost: " << TimeCostInMs(start, end) << endl;
    VLOG(VLOG_RUNTIME_DEBUG) << "Finish Plate Recognize Mxnet Processor : " << frameBatch->id() << endl;
    return true;
}

bool PlateRecognizeMxnetProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_PLATE, performance_);
}

bool PlateRecognizeMxnetProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    this->vehiclesFilter(frameBatch);
    return true;
}
void PlateRecognizeMxnetProcessor::setConfig(LPDRConfig_S *pstConfig) {
    readModuleFile(config_->fcnnSymbolFile, config_->fcnnParamFile,
                   &pstConfig->stFCNN, config_->is_model_encrypt);
    pstConfig->stFCNN.adwShape[0] = config_->batchsize;
    pstConfig->stFCNN.adwShape[1] = 1;    //channel
    pstConfig->stFCNN.adwShape[2] = config_->imageSH;  //standard height .
    pstConfig->stFCNN.adwShape[3] = config_->imageSW;  //standard width .

    readModuleFile(config_->rpnSymbolFile, config_->rpnParamFile,
                   &pstConfig->stRPN, config_->is_model_encrypt);
    pstConfig->stRPN.adwShape[0] = pstConfig->stFCNN.adwShape[0];
    pstConfig->stRPN.adwShape[1] = config_->numsPlates;//number of plates per image; .
    pstConfig->stRPN.adwShape[2] = 1;
    pstConfig->stRPN.adwShape[3] = config_->plateSH;// .
    pstConfig->stRPN.adwShape[4] = config_->plateSW;// .

    readModuleFile(config_->roipSymbolFile, config_->roipParamFile,
                   &pstConfig->stROIP, config_->is_model_encrypt);
    pstConfig->stROIP.adwShape[0] = pstConfig->stRPN.adwShape[0]
                                    * pstConfig->stRPN.adwShape[1];
    pstConfig->stROIP.adwShape[1] = 0;
    pstConfig->stROIP.adwShape[2] = 0;
    pstConfig->stROIP.adwShape[3] = 0;
    pstConfig->stROIP.adwShape[4] = pstConfig->stROIP.adwShape[0];
    pstConfig->stROIP.adwShape[5] = config_->numsProposal;//split to 20 small picture   proposal number of the image .
    pstConfig->stROIP.adwShape[6] = 5;

    readModuleFile(config_->pregSymbolFile, config_->pregParamFile,
                   &pstConfig->stPREG, config_->is_model_encrypt);
    pstConfig->stPREG.adwShape[0] = 1;
    pstConfig->stPREG.adwShape[1] = 1;
    pstConfig->stPREG.adwShape[2] = 64;
    pstConfig->stPREG.adwShape[3] = 64 * 2;

    readModuleFile(config_->chrecogSymbolFile, config_->chrecogParamFile,
                   &pstConfig->stCHRECOG, config_->is_model_encrypt);
    pstConfig->stCHRECOG.adwShape[0] = 50;
    pstConfig->stCHRECOG.adwShape[1] = 1;
    pstConfig->stCHRECOG.adwShape[2] = 32;
    pstConfig->stCHRECOG.adwShape[3] = 32;

    pstConfig->dwDevType = 2;
    pstConfig->dwDevID = config_->gpuId;

    batch_size_ = config_->batchsize;
    enable_local_province_ = config_->enableLocalProvince;
    local_province_ = config_->localProvinceText;
    local_province_confidence_ = config_->localProvinceConfidence;
}
void PlateRecognizeMxnetProcessor::vehiclesFilter(FrameBatch *frameBatch) {
    /*   images_.clear();
     images_.push_back(frameBatch->frames()[0]->payload()->data());
     return;*/
    images_.clear();
    objs_.clear();
    objs_ = frameBatch->CollectObjects(OPERATION_VEHICLE_PLATE);
    vector<Object *>::iterator itr = objs_.begin();
    while (itr != objs_.end()) {

        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {
            Vehicle *v = (Vehicle *) obj;
            VLOG(VLOG_RUNTIME_DEBUG) << "Put vehicle images to be color classified: " << obj->id() << endl;
            images_.push_back(v->image());
            ++itr;
            performance_++;

        } else {
            itr = objs_.erase(itr);
            VLOG(VLOG_RUNTIME_DEBUG) << "This is not a type of vehicle: " << obj->id() << endl;
        }
    }
}
}
/* namespace dg */
