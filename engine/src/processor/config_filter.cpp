/*
 * config_filter.cpp
 *
 *  Created on: May 6, 2016
 *      Author: jiajaichen
 */

#include "config_filter.h"

namespace dg {

ConfigFilter *ConfigFilter::instance_ = NULL;

ConfigFilter::ConfigFilter() {
}
void ConfigFilter::createFaceDetectorConfig(
    const Config &cconfig, FaceDetector::FaceDetectorConfig &config) {
    string model_path = (string) data_config_.Value(
        FILE_FACE_DETECT_MODEL_PATH);
    string trained_model = (string) data_config_.Value(
        FILE_FACE_DETECT_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(
        FILE_FACE_DETECT_DEPLOY_MODEL);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int batch_size = (int) cconfig.Value(ADVANCED_FACE_DETECT_BATCH_SIZE);
    int scale = (int) cconfig.Value(ADVANCED_FACE_DETECT_SCALE);
    float confidence = (float) cconfig.Value(ADVANCED_FACE_DETECT_CONFIDENCE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    config.model_file = model_path + trained_model;
    config.deploy_file = model_path + deploy_model;
    config.is_model_encrypt = is_encrypted;
    config.batch_size = batch_size;
    config.confidence = confidence;
    config.scale = scale;
    config.gpu_id = gpu_id;

}
void ConfigFilter::createFaceExtractorConfig(
    const Config &cconfig,
    FaceFeatureExtractor::FaceFeatureExtractorConfig &config) {
    string model_path = (string) data_config_.Value(
        FILE_FACE_EXTRACT_MODEL_PATH);
    string trained_model = (string) data_config_.Value(
        FILE_FACE_EXTRACT_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(
        FILE_FACE_EXTRACT_DEPLOY_MODEL);
    string align_model = (string) data_config_.Value(
        FILE_FACE_EXTRACT_ALIGN_MODEL);
    string align_deploy = (string) data_config_.Value(
        FILE_FACE_EXTRACT_ALIGN_DEPLOY);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int batch_size = (int) cconfig.Value(ADVANCED_FACE_EXTRACT_BATCH_SIZE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    config.model_file = model_path + trained_model;
    config.deploy_file = model_path + deploy_model;
    config.align_deploy = model_path + align_deploy;
    config.align_model = model_path + align_model;
    config.is_model_encrypt = is_encrypted;
    config.batch_size = batch_size;
    config.gpu_id = gpu_id;
}

void ConfigFilter::createVehicleConfig(
    const Config &cconfig,
    vector<VehicleCaffeClassifier::VehicleCaffeConfig> &configs) {
    string model_path = (string) data_config_.Value(FILE_STYLE_MODEL_PATH);
    string trained_model = (string) data_config_.Value(
        FILE_STYLE_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(FILE_STYLE_DEPLOY_MODEL);
    int batch_size = (int) cconfig.Value(ADVANCED_STYLE_BATCH_SIZE);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    int model_num = (int) cconfig.Value(ADVANCED_STYLE_MODEL_NUM);
    model_num = model_num == 0 ? 1 : model_num;

    for (int i = 0; i < model_num; i++) {
        VehicleCaffeClassifier::VehicleCaffeConfig config;
        config.model_file = model_path +  trained_model
            + to_string(i) + ".dat";
        config.deploy_file = model_path + deploy_model+to_string(i)+".txt";

        config.is_model_encrypt = is_encrypted;
        config.batch_size = batch_size;
        config.gpu_id = gpu_id;

        configs.push_back(config);
    }
}

void ConfigFilter::createVehicleColorConfig(
    const Config &cconfig,
    vector<CaffeVehicleColorClassifier::VehicleColorConfig> &configs) {
    string model_path = (string) data_config_.Value(FILE_COLOR_MODEL_PATH);
    string trained_model = (string) data_config_.Value(
        FILE_COLOR_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(FILE_COLOR_DEPLOY_MODEL);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int batch_size = (int) cconfig.Value(ADVANCED_COLOR_BATCH_SIZE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    int model_num = (int) cconfig.Value(ADVANCED_COLOR_MODEL_NUM);
    model_num = model_num == 0 ? 1 : model_num;
    for (int i = 0; i < model_num; i++) {
        CaffeVehicleColorClassifier ::VehicleColorConfig config;
        config.model_file = model_path + trained_model;
        config.deploy_file = model_path + deploy_model;
        config.is_model_encrypt = is_encrypted;
        config.batch_size = batch_size;
        config.gpu_id = gpu_id;
        configs.push_back(config);
    }
}

void ConfigFilter::createVehiclePlateConfig(
    const Config &cconfig, PlateRecognizer::PlateConfig &pConfig) {
    pConfig.LocalProvince = (const string &) cconfig.Value(
        ADVANCED_PLATE_LOCAL_PROVINCE);
    pConfig.OCR = (int) cconfig.Value(ADVANCED_PLATE_OCR);
    pConfig.PlateLocate = (int) cconfig.Value(ADVANCED_PLATE_LOCATE);
    pConfig.isSharpen = (bool) cconfig.Value(ADVANCED_PLATE_ENBALE_SHARPEN);
}

void ConfigFilter::createVehicleCaffeDetectorConfig(
    const Config &cconfig,
    VehicleCaffeDetector::VehicleCaffeDetectorConfig &config) {
    string model_path = (string) data_config_.Value(FILE_DETECTION_MODEL_PATH);
    string trained_model = (string) data_config_.Value(
        FILE_DETECTION_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(
        FILE_DETECTION_DEPLOY_MODEL);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int batch_size = (int) cconfig.Value(ADVANCED_DETECTION_BATCH_SIZE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    config.model_file = model_path + trained_model;
    config.deploy_file = model_path + deploy_model;
    config.is_model_encrypt = is_encrypted;
    config.batch_size = batch_size;
    config.gpu_id = gpu_id;

}
void ConfigFilter::createAccelerateConfig(
    const Config &cconfig,
    VehicleCaffeDetector::VehicleCaffeDetectorConfig &config) {
    string model_path = (string) data_config_.Value(FILE_ACCELERATE_MODEL_PATH);
    string trained_model = (string) data_config_.Value(
        FILE_ACCELERATE_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(
        FILE_ACCELERATE_DEPLOY_MODEL);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    config.model_file = model_path + trained_model;
    config.deploy_file = model_path + deploy_model;
    config.is_model_encrypt = is_encrypted;
    config.batch_size = 1;
    config.gpu_id = gpu_id;

}

//void ConfigFilter::createVehicleMutiTypeDetectorConfig(
//    const Config &cconfig,
//    VehicleMultiTypeDetector::VehicleMultiTypeConfig &config) {
//    string model_path = (string) data_config_.Value(FILE_DETECTION_MODEL_PATH);
//    string trained_model = (string) data_config_.Value(
//        FILE_DETECTION_TRAINED_MODEL);
//    string deploy_model = (string) data_config_.Value(
//        FILE_DETECTION_DEPLOY_MODEL);
//    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
//    int batch_size = (int) cconfig.Value(ADVANCED_DETECTION_BATCH_SIZE);
//    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
//
//    config.model_file = model_path + trained_model;
//    config.deploy_file = model_path + deploy_model;
//    config.is_model_encrypt = is_encrypted;
//    config.batch_size = batch_size;
//    config.gpu_id = gpu_id;
//
//}

void ConfigFilter::createMarkersConfig(
    const Config &cconfig, MarkerCaffeClassifier::MarkerConfig &mConfig) {

    int mot_confidence = (int) cconfig.Value(ADVANCED_MARKER_MOT_CONFIDENCE);
    int belt_confidence = (int) cconfig.Value(ADVANCED_MARKER_BETLT_CONFIDENCE);
    int global_confidence = (int) cconfig.Value(
        ADVANCED_MARKER_GLOBAL_CONFIDENCE);
    int accessories_confidence = (int) cconfig.Value(
        ADVANCED_MARKER_ACCESSORIES_CONFIDENCE);
    int others_confidence = (int) cconfig.Value(
        ADVANCED_MARKER_OTHERS_CONFIDENCE);
    int tissuebox_confidence = (int) cconfig.Value(
        ADVANCED_MARKER_TISSUEBOX_CONFIDENCE);
    int sunvisor_confidence = (int) cconfig.Value(
        ADVANCED_MARKER_SUNVISOR_CONFIDENCE);
    int batch_size = (int) cconfig.Value(ADVANCED_MARKER_BATCH_SIZE);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    mConfig.marker_confidence.insert(
        make_pair<int, float>(MarkerCaffeClassifier::MOT, mot_confidence));
    mConfig.marker_confidence.insert(
        make_pair<int, float>(MarkerCaffeClassifier::Belt,
                              belt_confidence));
    mConfig.marker_confidence.insert(
        make_pair<int, float>(MarkerCaffeClassifier::Global,
                              global_confidence));
    mConfig.marker_confidence.insert(
        make_pair<int, float>(MarkerCaffeClassifier::Accessories,
                              accessories_confidence));
    mConfig.marker_confidence.insert(
        make_pair<int, float>(MarkerCaffeClassifier::Others,
                              others_confidence));
    mConfig.marker_confidence.insert(
        make_pair<int, float>(MarkerCaffeClassifier::TissueBox,
                              tissuebox_confidence));
    mConfig.marker_confidence.insert(
        make_pair<int, float>(MarkerCaffeClassifier::SunVisor,
                              sunvisor_confidence));
    mConfig.model_file = (string) data_config_.Value(FILE_MARKER_MODEL_PATH)
        + (string) data_config_.Value(FILE_MARKER_TRAINED_MODEL);
    mConfig.deploy_file = (string) data_config_.Value(FILE_MARKER_MODEL_PATH)
        + (string) data_config_.Value(FILE_MARKER_DEPLOY_MODEL);

    mConfig.is_model_encrypt = is_encrypted;
    mConfig.batch_size = batch_size;
    mConfig.gpu_id = gpu_id;

}

void ConfigFilter::createWindowConfig(
    const Config &cconfig,
    WindowCaffeDetector::WindowCaffeConfig &wConfig) {
    int batch_size = (int) cconfig.Value(ADVANCED_WINDOW_BATCH_SIZE);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    wConfig.model_file = (string) data_config_.Value(FILE_WINDOW_MODEL_PATH)
        + (string) data_config_.Value(FILE_WINDOW_TRAINED_MODEL);
    wConfig.deploy_file = (string) data_config_.Value(FILE_WINDOW_MODEL_PATH)
        + (string) data_config_.Value(FILE_WINDOW_DEPLOY_MODEL);
    wConfig.is_model_encrypt = is_encrypted;
    wConfig.batch_size = batch_size;
    wConfig.gpu_id = gpu_id;

}
void ConfigFilter::createPlateMxnetConfig(const Config &cconfig, _LPDRConfig *pConfig) {
    pConfig->fcnnSymbolFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_FCN_SYMBOL);
    pConfig->fcnnParamFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_FCN_PARAM);

    pConfig->rpnSymbolFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_RPN_SYMBOL);
    pConfig->rpnParamFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_RPN_PARAM);

    pConfig->roipSymbolFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_ROIP_SYMBOL);
    pConfig->roipParamFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_ROIP_PARAM);

    pConfig->pregSymbolFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_POLYREG_SYMBOL);
    pConfig->pregParamFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_POLYREG_PARAM);

    pConfig->chrecogSymbolFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_CHRECOG_SYMBOL);
    pConfig->chrecogParamFile = (string) data_config_.Value(FILE_PLATE_MODEL_PATH)
        + (string) data_config_.Value(FILE_PLATE_CHRECOG_PARAM);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    pConfig->dwDevType=2;
    pConfig->dwDevID=gpu_id;
    pConfig->imageSW=(int)cconfig.Value(ADVANCED_PLATE_MXNET_IMGSW);
    pConfig->imageSH=(int)cconfig.Value(ADVANCED_PLATE_MXNET_IMGSH);
    pConfig->plateSW=(int)cconfig.Value(ADVANCED_PLATE_MXNET_PLATESW);
    pConfig->plateSH=(int)cconfig.Value(ADVANCED_PLATE_MXNET_PLATESH);
    pConfig->numsProposal=(int)cconfig.Value(ADVANCED_PLATE_MXNET_NUMSPROPOSAL);
    pConfig->numsPlates=(int)cconfig.Value(ADVANCED_PLATE_MXNET_PLATENUMS);
    int batch_size = (int) cconfig.Value(ADVANCED_PLATE_MXNET_BATCHSIZE);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    pConfig->is_model_encrypt=is_encrypted;

    pConfig->batchsize = batch_size;
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

