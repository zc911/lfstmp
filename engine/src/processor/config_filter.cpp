/*
 * config_filter.cpp
 *
 *  Created on: May 6, 2016
 *      Author: jiajaichen
 */

#include <alg/detector/detector.h>
#include "config_filter.h"

namespace dg {

ConfigFilter *ConfigFilter::instance_ = NULL;

ConfigFilter::ConfigFilter() {
}
void ConfigFilter::createFaceDetectorConfig(const Config &cconfig,
                                            FaceDetector::FaceDetectorConfig &config) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string model_path = (string) data_config_.Value(FILE_FACE_DETECT_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    string trained_model = (string) data_config_.Value(FILE_FACE_DETECT_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(FILE_FACE_DETECT_DEPLOY_MODEL);
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
void ConfigFilter::createFaceExtractorConfig(const Config &cconfig,
                                             FaceFeatureExtractor::FaceFeatureExtractorConfig &config) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string
        model_path = (string) data_config_.Value(FILE_FACE_EXTRACT_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    string trained_model = (string) data_config_.Value(FILE_FACE_EXTRACT_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(FILE_FACE_EXTRACT_DEPLOY_MODEL);
    string align_model = (string) data_config_.Value(FILE_FACE_EXTRACT_ALIGN_MODEL);
    string align_deploy = (string) data_config_.Value(FILE_FACE_EXTRACT_ALIGN_DEPLOY);

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

void ConfigFilter::createVehicleConfig(const Config &cconfig,
                                       vector<VehicleCaffeClassifier::VehicleCaffeConfig> &configs) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string model_path = (string) data_config_.Value(FILE_STYLE_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    string trained_model = (string) data_config_.Value(FILE_STYLE_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(FILE_STYLE_DEPLOY_MODEL);

    int batch_size = (int) cconfig.Value(ADVANCED_STYLE_BATCH_SIZE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    int model_num = (int) cconfig.Value(ADVANCED_STYLE_MODEL_NUM);
    model_num = model_num == 0 ? 1 : model_num;

    for (int i = 0; i < model_num; i++) {
        VehicleCaffeClassifier::VehicleCaffeConfig config;
        config.model_file = model_path + trained_model + std::to_string(i) + ".dat";
        config.deploy_file = model_path + deploy_model + std::to_string(i) + ".txt";

        config.is_model_encrypt = is_encrypted;
        config.batch_size = batch_size;
        config.gpu_id = gpu_id;

        configs.push_back(config);
    }
}

void ConfigFilter::createVehicleColorConfig(const Config &cconfig,
                                            vector<CaffeVehicleColorClassifier::VehicleColorConfig> &configs) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string model_path = (string) data_config_.Value(FILE_COLOR_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    string trained_model = (string) data_config_.Value(FILE_COLOR_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(FILE_COLOR_DEPLOY_MODEL);

    int batch_size = (int) cconfig.Value(ADVANCED_COLOR_BATCH_SIZE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    int model_num = (int) cconfig.Value(ADVANCED_COLOR_MODEL_NUM);
    model_num = model_num == 0 ? 1 : model_num;
    for (int i = 0; i < model_num; i++) {
        CaffeVehicleColorClassifier::VehicleColorConfig config;
        config.model_file = model_path + trained_model;
        config.deploy_file = model_path + deploy_model;
        config.is_model_encrypt = is_encrypted;
        config.batch_size = batch_size;
        config.gpu_id = gpu_id;
        configs.push_back(config);
    }
}

void ConfigFilter::createVehiclePlateConfig(const Config &cconfig,
                                            PlateRecognizer::PlateConfig &pConfig) {

    pConfig.LocalProvince = (const string &) cconfig.Value(ADVANCED_PLATE_LOCAL_PROVINCE);
    pConfig.OCR = (int) cconfig.Value(ADVANCED_PLATE_OCR);
    pConfig.PlateLocate = (int) cconfig.Value(ADVANCED_PLATE_LOCATE);
    pConfig.isSharpen = (bool) cconfig.Value(ADVANCED_PLATE_ENBALE_SHARPEN);
}

void ConfigFilter::createVehicleCaffeDetectorConfig(const Config &cconfig,
                                                    VehicleCaffeDetectorConfig &config) {
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    bool carOnly = (bool) cconfig.Value(ADVANCED_DETECTION_CAR_ONLY);

    string model_path = (string) data_config_.Value(FILE_DETECTION_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");
    string trained_model = (string) data_config_.Value(FILE_DETECTION_TRAINED_MODEL);

    string carOnlyPath =
        (string) data_config_.Value(FILE_CAR_ONLY_DETECTION_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");
    string carOnlyDeploy = (string) data_config_.Value(FILE_CAR_ONLY_DETECTION_DEPLOY_MODEL);
    string carOnlyModel = (string) data_config_.Value(FILE_CAR_ONLY_DETECTION_TRAINED_MODEL);

    string confirmPath =
        (string) data_config_.Value(FILE_CAR_ONLY_CONFIRM_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");
    string confirmDeploy = (string) data_config_.Value(FILE_CAR_ONLY_CONFIRM_DEPLOY_MODEL);
    string confirmModel = (string) data_config_.Value(FILE_CAR_ONLY_CONFIRM_TRAINED_MODEL);

    string deploy_model = (string) data_config_.Value(
        FILE_DETECTION_DEPLOY_MODEL);

    int batch_size = (int) cconfig.Value(ADVANCED_DETECTION_BATCH_SIZE);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    float minSize = (float) cconfig.Value(ADVANCED_DETECTION_TARGET_MIN_SIZE);
    float maxSize = (float) cconfig.Value(ADVANCED_DETECTION_TARGET_MAX_SIZE);


    config.target_min_size = minSize == 0 ? 400.0 : minSize;
    config.target_max_size = maxSize == 0 ? 600.0 : maxSize;
    config.car_only = carOnly;

    if (carOnly) {
        config.model_file = carOnlyPath + carOnlyModel;
        config.deploy_file = carOnlyPath + carOnlyDeploy;
        config.confirm_model_file = confirmPath + confirmModel;
        config.confirm_deploy_file = confirmPath + confirmDeploy;
    } else {
        config.model_file = model_path + trained_model;
        config.deploy_file = model_path + deploy_model;
        config.target_min_size=minSize;
        config.target_max_size=maxSize;
    }


    config.is_model_encrypt = is_encrypted;
    config.batch_size = batch_size;
    config.gpu_id = gpu_id;

}
void ConfigFilter::createAccelerateConfig(const Config &cconfig,
                                          VehicleCaffeDetectorConfig &config) {

    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    string model_path = (string) data_config_.Value(FILE_ACCELERATE_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");
    string trained_model = (string) data_config_.Value(
        FILE_ACCELERATE_TRAINED_MODEL);
    string deploy_model = (string) data_config_.Value(
        FILE_ACCELERATE_DEPLOY_MODEL);

    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    config.car_only = false;
    config.model_file = model_path + trained_model;
    config.deploy_file = model_path + deploy_model;
    config.is_model_encrypt = is_encrypted;
    config.batch_size = 1;
    config.gpu_id = gpu_id;
    config.target_min_size=4;
    config.target_max_size=6;

}


void ConfigFilter::createMarkersConfig(const Config &cconfig,
                                       MarkerCaffeClassifier::MarkerConfig &mConfig) {

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

    string modelPath = (string) data_config_.Value(FILE_MARKER_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");
    mConfig.model_file = modelPath + (string) data_config_.Value(FILE_MARKER_TRAINED_MODEL);
    mConfig.deploy_file = modelPath + (string) data_config_.Value(FILE_MARKER_DEPLOY_MODEL);

    mConfig.is_model_encrypt = is_encrypted;
    mConfig.batch_size = batch_size;
    mConfig.gpu_id = gpu_id;

}

void ConfigFilter::createPedestrianConfig(const Config &cconfig,
                                          PedestrianClassifier::PedestrianConfig &pConfig) {
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    string
        modelPath = (string) data_config_.Value(FILE_PEDESTRIAN_ATTR_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    pConfig.model_file = modelPath + (string) data_config_.Value(FILE_PEDESTRIAN_ATTR_TRAINED_MODEL);
    pConfig.deploy_file = modelPath + (string) data_config_.Value(FILE_PEDESTRIAN_ATTR_DEPLOY_MODEL);
    pConfig.tag_name_path = modelPath + (string) data_config_.Value(FILE_PEDESTRIAN_ATTR_TAGNAME_MODEL);
    pConfig.layer_name = (string) data_config_.Value(
        FILE_PEDESTRIAN_ATTR_LAYER_NAME);
    pConfig.gpu_id = gpu_id;
    pConfig.is_model_encrypt = is_encrypted;
}

void ConfigFilter::createWindowConfig(const Config &cconfig,
                                      WindowCaffeDetector::WindowCaffeConfig &wConfig) {
    int batch_size = (int) cconfig.Value(ADVANCED_WINDOW_BATCH_SIZE);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    string
        modelPath = (string) data_config_.Value(FILE_WINDOW_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    wConfig.model_file = modelPath
        + (string) data_config_.Value(FILE_WINDOW_TRAINED_MODEL);
    wConfig.deploy_file = modelPath
        + (string) data_config_.Value(FILE_WINDOW_DEPLOY_MODEL);
    wConfig.is_model_encrypt = is_encrypted;
    wConfig.batch_size = batch_size;
    wConfig.gpu_id = gpu_id;

}
void ConfigFilter::createMarkersConfig(const Config &cconfig,
                         VehicleCaffeDetectorConfig &mConfig){
    int batch_size = (int) cconfig.Value(ADVANCED_MARKER_BATCH_SIZE);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);


    string modelPath = (string) data_config_.Value(FILE_MARKER_ONLY_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");
    mConfig.model_file = modelPath + (string) data_config_.Value(FILE_MARKER_ONLY_TRAINED_MODEL);
    mConfig.deploy_file = modelPath + (string) data_config_.Value(FILE_MARKER_ONLY_DEPLOY_MODEL);

    mConfig.is_model_encrypt = is_encrypted;
    mConfig.batch_size = batch_size;
    mConfig.gpu_id = gpu_id;
}
void ConfigFilter::createWindowConfig(const Config &cconfig,
                        VehicleCaffeDetectorConfig &wConfig){
    int batch_size = (int) cconfig.Value(ADVANCED_WINDOW_BATCH_SIZE);
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);
    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);

    string
        modelPath = (string) data_config_.Value(FILE_WINDOW_ONLY_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    wConfig.model_file = modelPath
        + (string) data_config_.Value(FILE_WINDOW_ONLY_TRAINED_MODEL);
    wConfig.deploy_file = modelPath
        + (string) data_config_.Value(FILE_WINDOW_ONLY_DEPLOY_MODEL);
    wConfig.target_max_size=160;
    wConfig.target_min_size=80;
    wConfig.is_model_encrypt = is_encrypted;
    wConfig.batch_size = batch_size;
    wConfig.gpu_id = gpu_id;
}
void ConfigFilter::createPlateMxnetConfig(const Config &cconfig,PlateRecognizeMxnetProcessor::PlateRecognizeMxnetConfig *pConfig) {
    bool is_encrypted = (bool) cconfig.Value(DEBUG_MODEL_ENCRYPT);

    pConfig->is_model_encrypt = is_encrypted;

    string
        modelPath = (string) data_config_.Value(FILE_PLATE_MODEL_PATH) + (is_encrypted == true ? "1/" : "0/");

    pConfig->fcnnSymbolFile = modelPath + (string) data_config_.Value(FILE_PLATE_FCN_SYMBOL);
    pConfig->fcnnParamFile = modelPath + (string) data_config_.Value(FILE_PLATE_FCN_PARAM);

    pConfig->rpnSymbolFile = modelPath + (string) data_config_.Value(FILE_PLATE_RPN_SYMBOL);
    pConfig->rpnParamFile = modelPath + (string) data_config_.Value(FILE_PLATE_RPN_PARAM);

    pConfig->roipSymbolFile = modelPath + (string) data_config_.Value(FILE_PLATE_ROIP_SYMBOL);
    pConfig->roipParamFile = modelPath + (string) data_config_.Value(FILE_PLATE_ROIP_PARAM);

    pConfig->pregSymbolFile = modelPath + (string) data_config_.Value(FILE_PLATE_POLYREG_SYMBOL);
    pConfig->pregParamFile = modelPath + (string) data_config_.Value(FILE_PLATE_POLYREG_PARAM);

    pConfig->chrecogSymbolFile = modelPath + (string) data_config_.Value(FILE_PLATE_CHRECOG_SYMBOL);
    pConfig->chrecogParamFile = modelPath + (string) data_config_.Value(FILE_PLATE_CHRECOG_PARAM);

    int gpu_id = (int) cconfig.Value(SYSTEM_GPUID);
    pConfig->gpuId = gpu_id;
    pConfig->imageSW = (int) cconfig.Value(ADVANCED_PLATE_MXNET_IMGSW);
    pConfig->imageSH = (int) cconfig.Value(ADVANCED_PLATE_MXNET_IMGSH);
    pConfig->plateSW = (int) cconfig.Value(ADVANCED_PLATE_MXNET_PLATESW);
    pConfig->plateSH = (int) cconfig.Value(ADVANCED_PLATE_MXNET_PLATESH);
    pConfig->numsProposal = (int) cconfig.Value(ADVANCED_PLATE_MXNET_NUMSPROPOSAL);
    pConfig->numsPlates = (int) cconfig.Value(ADVANCED_PLATE_MXNET_PLATENUMS);
    pConfig->enableLocalProvince=(bool)cconfig.Value(ADVANCED_PLATE_MXNET_ENABLE_LOCALPROVINCE);
    pConfig->localProvinceText=(string)cconfig.Value(ADVANCED_PLATE_MXNET_LOCALPROVINCE_TEXT);
    pConfig->localProvinceConfidence=(float)cconfig.Value(ADVANCED_PLATE_MXNET_LOCALPROVINCE_CONFIDENCE);


    int batch_size = (int) cconfig.Value(ADVANCED_PLATE_MXNET_BATCHSIZE);


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

