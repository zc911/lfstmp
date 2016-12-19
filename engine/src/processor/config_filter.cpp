/*
 * config_filter.cpp
 *
 *  Created on: May 6, 2016
 *      Author: jiajaichen
 */

#include "config_filter.h"
#include "face_alignment_processor.h"
#include "face_quality_processor.h"

namespace dg {

ConfigFilter *ConfigFilter::instance_ = NULL;

ConfigFilter::ConfigFilter() {
}

void ConfigFilter::createFaceDetectorConfig(const Config &cconfig,
                                            FaceDetectorConfig &config) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string model_path = (string) data_config_.Value(FILE_FACE_DETECT_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    string trained_model = (string) data_config_.Value(FILE_FACE_DETECT_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(FILE_FACE_DETECT_DEPLOY_MODEL);
    int batch_size = (int) cconfig.Value(ADVANCED_FACE_DETECT_BATCH_SIZE);
    int scale = (int) cconfig.Value(ADVANCED_FACE_DETECT_SCALE);
    float confidence = (float) cconfig.Value(ADVANCED_FACE_DETECT_CONFIDENCE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    int method = (int) cconfig.Value(ADVANCED_FACE_DETECT_METHOD);
    config.model_file = model_path + trained_model;
    config.deploy_file = model_path + deploy_model;

    config.is_model_encrypt = is_encrypted;
    config.batch_size = batch_size;
    config.confidence = confidence;
    config.img_scale_min = (int) cconfig.Value(ADVANCED_FACE_DETECT_MIN);
    config.img_scale_max = (int) cconfig.Value(ADVANCED_FACE_DETECT_MAX);

    config.scale = scale;
    config.gpu_id = gpu_id;

}
void ConfigFilter::createFaceQualityConfig(const Config &cconfig, FaceQualityConfig &fqConfig) {
    fqConfig.blur_threshold = (float) cconfig.Value(ADVANCED_FACE_QUALITY_THRESHOLD);
}

void ConfigFilter::createFaceAlignmentConfig(const Config &cconfig, FaceAlignmentConfig &faConfig) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string
        model_path = (string) data_config_.Value(FILE_FACE_EXTRACT_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    int detect_method = (int) cconfig.Value(ADVANCED_FACE_DETECT_METHOD);

    string align_model = (string) data_config_.Value(FILE_FACE_EXTRACT_ALIGN_MODEL);
    string align_deploy = (string) data_config_.Value(FILE_FACE_EXTRACT_ALIGN_DEPLOY);
    string align_config = (string) data_config_.Value(FILE_FACE_EXTRACT_ALIGN_CONFIG);
    string align_path = (string) data_config_.Value(FILE_FACE_EXTRACT_ALIGN_PATH);
    int face_size_num = (int) cconfig.Value(ADVANCED_FACE_EXTRACT_ALIGNMENT_FACESIZE + "/Size");

    vector<int> face_size;
    for (int i = 0; i < face_size_num; i++) {
        face_size.push_back((int) cconfig.Value(ADVANCED_FACE_EXTRACT_ALIGNMENT_FACESIZE + to_string(i)));
    }

    faConfig.align_deploy = model_path + align_deploy;
    faConfig.align_cfg = model_path + align_config;
    faConfig.method = (int) cconfig.Value(ADVANCED_FACE_ALIGN_METHOD);
    faConfig.gpu_id = gpu_id;
    faConfig.threshold = (float) cconfig.Value(ADVANCED_FACE_ALIGN_THRESHOLD);
    faConfig.align_path = model_path + align_path;
    faConfig.align_model = model_path;
    faConfig.align_model_path = model_path;


    faConfig.method = 2;
    faConfig.is_model_encrypt = is_encrypted;
    faConfig.face_size = face_size;
}


void ConfigFilter::createFaceExtractorConfig(const Config &cconfig,
                                             FaceFeatureExtractorConfig &config) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string
        model_path = (string) data_config_.Value(FILE_FACE_EXTRACT_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    string trained_model = (string) data_config_.Value(FILE_FACE_EXTRACT_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(FILE_FACE_EXTRACT_DEPLOY_MODEL);
    string model_dir = (string) data_config_.Value(FILE_FACE_EXTRACT_MODEL_DIR);


    int batch_size = (int) cconfig.Value(ADVANCED_FACE_EXTRACT_BATCH_SIZE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    int face_size_num = (int) cconfig.Value(ADVANCED_FACE_EXTRACT_ALIGNMENT_FACESIZE + "/Size");
    int mean_size = (int) data_config_.Value(FILE_FACE_EXTRACT_MEAN + "/Size");
    vector<int> face_size;
    for (int i = 0; i < face_size_num; i++) {
        face_size.push_back((int) cconfig.Value(ADVANCED_FACE_EXTRACT_ALIGNMENT_FACESIZE + to_string(i)));
    }
    config.model_file = model_path + trained_model;
    config.deploy_file = model_path + deploy_model;
    config.is_model_encrypt = is_encrypted;
    config.batch_size = batch_size;
    config.gpu_id = gpu_id;
    config.layer_name = (string) data_config_.Value(FILE_FACE_EXTRACT_LAYERNAME);
    config.islog = (bool) cconfig.Value(DEBUG_ENABLE);


    for (int i = 0; i < mean_size; i++) {
        config.mean.push_back((int) data_config_.Value(FILE_FACE_EXTRACT_MEAN + to_string(i)));
    }
    config.pixel_scale = (float) data_config_.Value(FILE_FACE_EXTRACT_PIXEL_SCALE);
    config.face_size = face_size;
    config.pre_process = (string) cconfig.Value(ADVANCED_FACE_EXTRACT_PRE_PROCESS);
    config.use_GPU = true;
    config.method = (int) cconfig.Value(ADVANCED_FACE_EXTRACT_METHOD);
    config.model_config =
        model_path + (string) data_config_.Value(FILE_FACE_EXTRACT_MODEL_CONFIG) + to_string(config.method) + ".cfg";
    config.model_dir = model_path + model_dir;
    config.method = (int) cconfig.Value(ADVANCED_FACE_EXTRACT_METHOD);
    config.concurrency = (bool) cconfig.Value(ADVANCED_FACE_ENABLE_CONCURRENCY);

}

void ConfigFilter::createVehiclePlateConfig(const Config &cconfig,
                                            PlateRecognizer::PlateConfig &pConfig) {

    pConfig.LocalProvince = (const string &) cconfig.Value(ADVANCED_PLATE_LOCAL_PROVINCE);
    pConfig.OCR = (int) cconfig.Value(ADVANCED_PLATE_OCR);
    pConfig.PlateLocate = (int) cconfig.Value(ADVANCED_PLATE_LOCATE);
    pConfig.isSharpen = (bool) cconfig.Value(ADVANCED_PLATE_ENBALE_SHARPEN);
}

void ConfigFilter::createPlateMxnetConfig(const Config &cconfig,
                                          PlateRecognizeMxnetProcessor::PlateRecognizeMxnetConfig& pConfig) {
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

