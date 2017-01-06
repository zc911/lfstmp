/*
 * config_filter.cpp
 *
 *  Created on: May 6, 2016
 *      Author: jiajaichen
 */

#include "config_filter.h"
#include "face_alignment_processor.h"

namespace dg {

ConfigFilter *ConfigFilter::instance_ = NULL;

ConfigFilter::ConfigFilter() {
}

void ConfigFilter::createFaceDetectorConfig(const Config &cconfig,
                                            FaceDetectorConfig &config) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string model_path = (string) cconfig.Value(DGFACE_MODEL_PATH) + "/dgface.json";
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    config.is_model_encrypt = is_encrypted;
    config.model_dir = model_path;
    config.gpu_id = gpu_id;

}
void ConfigFilter::createFaceQualityConfig(const Config &cconfig, FaceQualityConfig &fqConfig) {
    fqConfig.blur_threshold = (float) cconfig.Value(ADVANCED_FACE_QUALITY_THRESHOLD);
}

void ConfigFilter::createFaceAlignmentConfig(const Config &cconfig, FaceAlignmentConfig &faConfig) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);

    string model_path = (string) cconfig.Value(DGFACE_MODEL_PATH) + "/dgface.json";
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    float alignThreshold = (float) cconfig.Value(ADVANCED_FACE_ALIGN_THRESHOLD);
    faConfig.align_threshold = alignThreshold;
    faConfig.gpu_id = gpu_id;
    faConfig.model_dir = model_path;

    faConfig.is_model_encrypt = is_encrypted;
}


void ConfigFilter::createFaceExtractorConfig(const Config &cconfig,
                                             FaceFeatureExtractorConfig &config) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    string model_path = (string) cconfig.Value(DGFACE_MODEL_PATH) + "/dgface.json";
    config.is_model_encrypt = is_encrypted;
    config.gpu_id = gpu_id;
    config.use_GPU = true;
    config.concurrency = (bool) cconfig.Value(ADVANCED_FACE_ENABLE_CONCURRENCY);
    config.model_dir = model_path;

}

//void ConfigFilter::createVehiclePlateConfig(const Config &cconfig,
//                                            PlateRecognizer::PlateConfig &pConfig) {
//
//    pConfig.LocalProvince = (const string &) cconfig.Value(ADVANCED_PLATE_LOCAL_PROVINCE);
//    pConfig.OCR = (int) cconfig.Value(ADVANCED_PLATE_OCR);
//    pConfig.PlateLocate = (int) cconfig.Value(ADVANCED_PLATE_LOCATE);
//    pConfig.isSharpen = (bool) cconfig.Value(ADVANCED_PLATE_ENBALE_SHARPEN);
//}

void ConfigFilter::createPlateMxnetConfig(const Config &cconfig,
                                          PlateRecognizeMxnetProcessor::PlateRecognizeMxnetConfig &pConfig) {
    pConfig.gpuId = (int) cconfig.Value(SYSTEM_GPUID);
    pConfig.is_model_encrypt = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    pConfig.modelPath = (string) cconfig.Value(DGLP_MODEL_PATH);
    pConfig.enableLocalProvince = (bool) cconfig.Value(ADVANCED_PLATE_MXNET_ENABLE_LOCALPROVINCE);
    pConfig.localProvinceText = (string) cconfig.Value(ADVANCED_PLATE_MXNET_LOCALPROVINCE_TEXT);
    pConfig.localProvinceConfidence = (float) cconfig.Value(ADVANCED_PLATE_MXNET_LOCALPROVINCE_CONFIDENCE);
    pConfig.batchsize = (int) cconfig.Value(ADVANCED_PLATE_MXNET_BATCHSIZE);
}

int ConfigFilter::initDataConfig(const Config &config) {
    string data_config_path = (string) config.Value(DATAPATH);
    string json_data = ReadStringFromFile(data_config_path, "r");
#ifndef DEBUG
    //TODO: decrypted from file
#endif
    data_config_.LoadString(json_data);
    return 1;
}

}

