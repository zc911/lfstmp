/*============================================================================
 * File Name   : witness_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_WITNESS_SERVICE_H_
#define MATRIX_APPS_WITNESS_SERVICE_H_

#include <mutex>
#include "config.h"
#include "matrix_engine/model/model.h"
#include "matrix_engine/engine/witness_engine.h"
#include "witness.grpc.pb.h"
#include "string_util.h"
#include "../config/config_val.h"
#include "grpc/spring_grpc.h"
#include "engine_service.h"
#include "witness_bucket.h"
#include "repo_service.h"


namespace dg {

using namespace ::dg::model;


class WitnessAppsService: public EngineService {
public:
    WitnessAppsService(Config *config, string name, int baseId = 0);
    virtual ~WitnessAppsService();

    MatrixError Recognize(const WitnessRequest *request, WitnessResponse *response);

    MatrixError BatchRecognize(const WitnessBatchRequest *request,
                               WitnessBatchResponse *response);

    string name_;
private:
    MatrixError getRecognizedVehicle(const Vehicle *vobj,
                                     RecVehicle *vrec);
    MatrixError getRecognizedPedestrian(const Pedestrian *pobj,
                                        RecPedestrian *vrec);
    MatrixError getRecognizedFace(const vector<const Face *>faceVector,
                                  ::google::protobuf::RepeatedPtrField< ::dg::model::RecPedestrian >* recPedestrian);
    MatrixError getRecognizeResult(Frame *frame, WitnessResult *result);

    MatrixError checkRequest(const WitnessRequest &request);
    MatrixError checkRequest(const WitnessBatchRequest &requests);
    MatrixError checkWitnessImage(const WitnessImage &wImage);


    Config *config_;
    Identification id_;
    Identification base_id_;


    // library caffe is not thread safe(even crash some times) which means
    // only one frame/frameBatch could be processed at one time.
    // So we use a lock to keep the processing thread safe.
    std::mutex rec_lock_;

    void init(void);
    bool enableStorage_;
    string storage_address_;
    bool enable_cutboard_;
    unsigned int parse_image_timeout_;

    static string trimString(string str);
    static int parseInt(string str);
    static Operation getOperation(const WitnessRequestContext &ctx);
    static void copyCutboard(const Detection &d, Cutboard *cb);

};

}

#endif //MATRIX_APPS_WITNESS_SERVICE_H_
