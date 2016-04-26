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

}

#endif /* ENGINE_CONFIG_VALUE_H_ */
