/*
 * plate_recognize_mxnet_processor.h
 *
 *  Created on: May 18, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_PLATE_RECOGNIZE_MXNET_PROCESSOR_H_
#define SRC_PROCESSOR_PLATE_RECOGNIZE_MXNET_PROCESSOR_H_

#include "processor/processor.h"
#include "processor_helper.h"
#include "LPDR.hpp"

namespace dg {
class PlateRecognizeMxnetProcessor: public Processor {

 public:
    typedef struct {
        string modelPath;
        int gpuId;
        bool is_model_encrypt = true;
        int batchsize = 1;
        bool enableLocalProvince;
        string localProvinceText;
        float localProvinceConfidence;
    } PlateRecognizeMxnetConfig;

    PlateRecognizeMxnetProcessor(const PlateRecognizeMxnetConfig &config);
    virtual ~PlateRecognizeMxnetProcessor();
 protected:
    virtual bool process(Frame *frame);
    virtual bool process(FrameBatch *frameBatch);

    virtual bool RecordFeaturePerformance();

    virtual bool beforeUpdate(FrameBatch *frameBatch);
 private:
    void vehiclesFilter(FrameBatch *frameBatch);

//    void setConfig(LPDRConfig_S *pstConfig);
//    PlateRecognizeMxnetConfig *config_ = NULL;
//    LPDR_HANDLE h_LPDR_Handle_ = 0;
    dgLP::LPDR *pclsLPDR;
    vector<Object *> objs_;
    LPDR_ImageSet_S stImgSet_;
    vector<Mat> images_;
    int batch_size_;
    bool enable_local_province_;
    string local_province_;
    float local_province_confidence_;
};

} /* namespace dg */

#endif /* SRC_PROCESSOR_PLATE_RECOGNIZE_MXNET_PROCESSOR_H_ */
