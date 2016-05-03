/*
 * engine_config_value.h
 *
 *  Created on: Apr 25, 2016
 *      Author: chenzhen
 */

#ifndef ENGINE_CONFIG_VALUE_H_
#define ENGINE_CONFIG_VALUE_H_

#include <string>

using namespace std;

namespace dg {

static const string SYSTEM_GPUID = "System/GpuId";

// feature/vehicle
static const string FEATURE_VEHICLE_ENABLE = "Feature/Vehicle/Enable";
static const string FEATURE_VEHICLE_ENABLE_DETECTION =
        "Feature/Vehicle/EnableDetection";
static const string FEATURE_VEHICLE_ENABLE_TYPE = "Feature/Vehicle/EnableType";
static const string FEATURE_VEHICLE_ENABLE_COLOR = "Feature/Vehicle/EnableColor";
static const string FEATURE_VEHICLE_ENABLE_PLATE = "Feature/Vehicle/EnablePlate";
static const string FEATURE_VEHICLE_ENABLE_PLATE_ENHANCED =
        "Feature/Vehicle/EnablePlateEnhance";
static const string FEATURE_VEHICLE_ENABLE_MARKER =
        "Feature/Vehicle/EnableMarker";
static const string FEATURE_VEHICLE_ENABLE_FEATURE_VECTOR =
        "Feature/Vehicle/EnableFeatureVector";
// feature/face
static const string FEATURE_FACE_ENABLE = "Feature/Face/Enable";
static const string FEATURE_FACE_ENABLE_DETECTION =
        "Feature/Face/EnableDetection";
static const string FEATURE_FACE_ENABLE_FEATURE_VECTOR =
        "Feature/Face/EnableFeatureVector";

static const string DEBUG_MODEL_ENCRYPT = "Debug/Encrypt";
static const string DEBUG_ENABLE = "Debug/Enable";

static const string ADVANCED_DETECTION_RESCALE = "Advanced/DetectionRescale";
static const string ADVANCED_DETECTION_BATCH_SIZE =
        "Advanced/DetectionBatchSize";
static const string ADVANCED_COLOR_BATCH_SIZE = "Advanced/ColorBatchSize";
static const string ADVANCED_STYLE_BATCH_SIZE = "Advanced/StyleBatchSize";
static const string ADVANCED_PLATE_LOCAL_PROVINCE =
        "Advanced/Plate/LocalProvince";
static const string ADVANCED_PLATE_OCR = "Advanced/Plate/OCR";
static const string ADVANCED_PLATE_LOCATE = "Advanced/Plate/Locate";

static const string FILE_COLOR_MODEL_PATH = "File/Color/ModelPath";
static const string FILE_COLOR_TRAINED_MODEL = "File/Color/TrainedModel";
static const string FILE_COLOR_DEPLOY_MODEL = "File/Color/DeployModel";

static const string FILE_STYLE_MODEL_PATH = "File/Style/ModelPath";
static const string FILE_STYLE_TRAINED_MODEL = "File/Style/TrainedModel";
static const string FILE_STYLE_DEPLOY_MODEL = "File/Style/DeployModel";

}

#endif /* ENGINE_CONFIG_VALUE_H_ */
