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

static const string ADVANCED_DETECTION_RESCALE = "Advanced/Detection/Rescale";
static const string ADVANCED_DETECTION_BATCH_SIZE =
        "Advanced/Detection/BatchSize";
static const string ADVANCED_COLOR_BATCH_SIZE = "Advanced/Color/BatchSize";
static const string ADVANCED_STYLE_BATCH_SIZE = "Advanced/Style/BatchSize";
static const string ADVANCED_WINDOW_BATCH_SIZE = "Advanced/Window/BatchSize";

static const string ADVANCED_MARKER_BATCH_SIZE = "Advanced/Marker/BatchSize";
static const string ADVANCED_MARKER_MOT_CONFIDENCE = "Advanced/Marker/MOTConfidence";
static const string ADVANCED_MARKER_BETLT_CONFIDENCE = "Advanced/Marker/BeltConfidence";
static const string ADVANCED_MARKER_GLOBAL_CONFIDENCE = "Advanced/Marker/GlobalConfidence";
static const string ADVANCED_MARKER_ACCESSORIES_CONFIDENCE = "Advanced/Marker/AccessoriesConfidence";
static const string ADVANCED_MARKER_OTHERS_CONFIDENCE = "Advanced/Marker/OthersConfidence";
static const string ADVANCED_MARKER_TISSUEBOX_CONFIDENCE = "Advanced/Marker/TissueBoxConfidence";
static const string ADVANCED_MARKER_SUNVISOR_CONFIDENCE = "Advanced/Marker/SunVisorConfidence";


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

static const string FILE_DETECTION_MODEL_PATH = "File/Detection/ModelPath";
static const string FILE_DETECTION_TRAINED_MODEL = "File/Detection/TrainedModel";
static const string FILE_DETECTION_DEPLOY_MODEL = "File/Detection/DeployModel";

static const string FILE_MARKER_MODEL_PATH = "File/Marker/ModelPath";
static const string FILE_MARKER_TRAINED_MODEL = "File/Marker/TrainedModel";
static const string FILE_MARKER_DEPLOY_MODEL = "File/Marker/DeployModel";

static const string FILE_WINDOW_MODEL_PATH = "File/Window/ModelPath";
static const string FILE_WINDOW_TRAINED_MODEL = "File/Window/TrainedModel";
static const string FILE_WINDOW_DEPLOY_MODEL = "File/Window/DeployModel";

}

#endif /* ENGINE_CONFIG_VALUE_H_ */
