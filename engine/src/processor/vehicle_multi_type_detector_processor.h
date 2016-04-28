/*
 * vehicle_detector_processor.h
 *
 *  Created on: 13/04/2016
 *      Author: chenzhen
 */

#ifndef VEHICLE_DETECTOR_PROCESSOR_H_
#define VEHICLE_DETECTOR_PROCESSOR_H_

#include <glog/logging.h>
#include "processor.h"
#include "alg/vehicle_multi_type_detector.h"
#include "util/debug_util.h"

namespace dg {

const static string MODEL_FILE =
        "models/detector/googlenet_faster_rcnn_iter_350000.caffemodel";
const static string DEPLOY_FILE = "models/detector/test.prototxt";
const static string MODEL_FILE_EN =
        "models/detector/googlenet_faster_rcnn_iter_350000.caffemodel";
const static string DEPLOY_FILE_EN = "models/detector/test.prototxt";

class VehicleMultiTypeDetectorProcessor : public Processor {
 public:

    VehicleMultiTypeDetectorProcessor(int batch_size, int gpu_id, int rescale,
                                      bool is_model_encrypt);

    ~VehicleMultiTypeDetectorProcessor();

    void Update(FrameBatch *frameBatch);

    bool checkOperation(Frame *frame);
    bool checkStatus(Frame *frame);

 private:
    VehicleMultiTypeDetector *detector_;

};

}
#endif /* VEHICLE_DETECTOR_PROCESSOR_H_ */
