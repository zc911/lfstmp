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

class EngineConfigValue {

 public:
    const static string SYSTEM_GPUID = "System/GpuId";

    // feature/vehicle
    const static string FEATURE_VEHICLE_ENABLE = "Feature/Vehicle/Enable";
    const static string FEATURE_VEHICLE_ENABLE_DETECTION =
            "Feature/Vehicle/EnableDetection";
    const static string FEATURE_VEHICLE_ENABLE_TYPE =
            "Feature/Vehicle/EnableType";
    const static string FEATURE_VEHICLE_ENABLE_COLOR =
            "Feature/Vehicle/EnableColor";
    const static string FEATURE_VEHICLE_ENABLE_PLATE =
            "Feature/Vehicle/EnablePlate";
    const static string FEATURE_VEHICLE_ENABLE_PLATE_ENHANCED =
            "Feature/Vehicle/EnablePlateEnhance";
    const static string FEATURE_VEHICLE_ENABLE_MARKER =
            "Feature/Vehicle/EnableMarker";
    const static string FEATURE_VEHICLE_ENABLE_FEATURE_VECTOR =
            "Feature/Vehicle/EnableFeatureVector";
    // feature/face
    const static string FEATURE_FACE_ENABLE = "Feature/Face/Enable";
    const static string FEATURE_FACE_ENABLE_DETECTION =
            "Feature/Face/EnableDetection";
    const static string FEATURE_FACE_ENABLE_FEATURE_VECTOR =
            "Feature/Face/EnableFeatureVector";

    const static string DEBUG_MODEL_ENCRYPT = "Debug/Encrypt";
    const static string DEBUG_ENABLE = "Debug/Enable";

    const static string ADVANCED_DETECTION_RESCALE = "Advanced/DetectionRescale";
    const static string ADVANCED_DETECTION_BATCH_SIZE =
            "Advanced/DetectionBatchSize";

};
}

#endif /* ENGINE_CONFIG_VALUE_H_ */
