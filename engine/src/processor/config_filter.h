/*
 * config_filter.h
 *
 *  Created on: May 5, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_ENGINE_CONFIG_FILTER_H_
#define SRC_ENGINE_CONFIG_FILTER_H_

#include "processor/processor.h"
#include "engine/engine_config_value.h"
#include "processor/vehicle_multi_type_detector_processor.h"
#include "processor/vehicle_classifier_processor.h"
#include "processor/vehicle_color_processor.h"
#include "processor/vehicle_marker_classifier_processor.h"
#include "processor/vehicle_plate_recognizer_processor.h"
#include "processor/car_feature_extract_processor.h"
#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"
#include "processor/pedestrian_classifier_processor.h"
#include "processor/vehicle_phone_detector_processor.h"
#include "processor/non_motor_vehicle_classifier_processor.h"
#include "plate_recognize_mxnet_processor.h"

#include "model/alg_config.h"
#include "config.h"
#include "fs_util.h"

namespace dg {
class ConfigFilter {

public:

  static ConfigFilter *GetInstance() {
    if (!instance_)
      instance_ = new ConfigFilter;
    return instance_;
  }

  void createVehiclePlateConfig(const Config &cconfig,
                                PlateRecognizer::PlateConfig &pConfig);
  void createPlateMxnetConfig(const Config &cconfig, PlateRecognizeMxnetProcessor::PlateRecognizeMxnetConfig *pConfig);

//  void createDriverBeltConfig(const Config &cconfig,
//                              VehicleBeltConfig &bConfig);
//  void createCoDriverBeltConfig(const Config &cconfig,
//                                VehicleBeltConfig &cbConfig);
//  void createVehicleCaffeDetectorConfig(const Config &cconfig,
//                                        VehicleCaffeDetectorConfig &config);
//  void createAccelerateConfig(const Config &cconfig,
//                              VehicleCaffeDetectorConfig &config);
 // void createFaceDetectorConfig(const Config &cconfig,
 //                               FaceDetector::FaceDetectorConfig &config);
 // void createFaceExtractorConfig(const Config &cconfig,
 //                                FaceFeatureExtractor::FaceFeatureExtractorConfig &config);
 // void createVehicleConfig(const Config &cconfig,
 //                          vector<VehicleCaffeClassifier::VehicleCaffeConfig> &configs);
  //void createVehicleColorConfig(const Config &cconfig, vector<CaffeVehicleColorClassifier::VehicleColorConfig> &configs);

//    void createVehicleMutiTypeDetectorConfig(
//        const Config &cconfig,
//        VehicleMultiTypeDetector::VehicleMultiTypeConfig &config);
//  void createMarkersConfig(const Config &cconfig, MarkerCaffeClassifier::MarkerConfig &mConfig);
//  void createWindowConfig(const Config &cconfig,
//                          WindowCaffeDetector::WindowCaffeConfig &wConfig);
//  void createMarkersConfig(const Config &cconfig,
//                           VehicleCaffeDetectorConfig &mConfig);

//  void createDriverPhoneConfig(const Config &cconfig,
//                               VehicleCaffeDetectorConfig &pConfig);
//  void createWindowConfig(const Config &cconfig,
//                          VehicleCaffeDetectorConfig &wConfig);
//  void createPedestrianConfig(const Config &cconfig, PedestrianClassifier::PedestrianConfig &pConfig);
//  void createPedestrianConfig(const Config &cconfig, NonMotorVehicleClassifier::NonMotorVehicleConfig &nmConfig);
  int initDataConfig(const Config &config);

private:
  ConfigFilter();
  static ConfigFilter *instance_;

  Config data_config_;
};

}

#endif /* SRC_ENGINE_CONFIG_FILTER_H_ */
