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
#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"
#include "processor/face_quality_processor.h"
#include "plate_recognize_mxnet_processor.h"
#include "processor/face_alignment_processor.h"
#include "plate_recognizer.h"
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

    void createFaceDetectorConfig(const Config &cconfig,
                                  FaceDetectorConfig &config);
    void createFaceQualityConfig(const Config &cconfig, FaceQualityConfig &fqConfig);

    void createFaceAlignmentConfig(const Config &cconfig,
                                   FaceAlignmentConfig &faConfig);

    void createFaceExtractorConfig(const Config &cconfig,
                                   FaceFeatureExtractorConfig &config);

    void createVehiclePlateConfig(const Config &cconfig,
                                PlateRecognizer::PlateConfig &pConfig);
    void createPlateMxnetConfig(const Config &cconfig, PlateRecognizeMxnetProcessor::PlateRecognizeMxnetConfig& pConfig);

    int initDataConfig(const Config &config);

private:
  ConfigFilter();
  static ConfigFilter *instance_;

  Config data_config_;
};

}

#endif /* SRC_ENGINE_CONFIG_FILTER_H_ */
