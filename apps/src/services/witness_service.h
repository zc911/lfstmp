/*============================================================================
 * File Name   : ranker_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_WITNESS_SERVICE_H_
#define MATRIX_APPS_WITNESS_SERVICE_H_

#include "config.h"
#include "model/witness.grpc.pb.h"
#include "engine/witness_engine.h"

namespace dg 
{
using namespace ::dg::model;

class WitnessAppsService
{
public:
    WitnessAppsService(const Config *config);
    virtual ~WitnessAppsService();

    bool Recognize(const WitnessRequest *request, WitnessResponse *response);

    bool BatchRecognize(const WitnessBatchRequest *request, WitnessBatchResponse *response);

private:
    const Config *config_;
    WitnessEngine engine_;
    Identification id_;

    static Operation getOperation(const WitnessRequestContext& ctx);
    static void copyCutboard(const Box &b, Cutboard *cb);

    MatrixError fillModel(Identification id, VehicleModel *model);
    MatrixError fillColor(const dg::Color &color, model::Color *rcolor);
    MatrixError fillPlate(const Plate &plate, LicensePlate *rplate);
    MatrixError fillSymbols(const vector<Object*>& objects, RecognizedVehicle *vrec));
    MatrixError getRecognizedVehicle(Vehicle *vobj, RecognizedVehicle *vrec);
    MatrixError getRecognizedFace(Face *fobj, RecognizedFace *frec);
    MatrixError getRecognizeResult(const Frame *frame, WitnessResult *result);
};

}

#endif //MATRIX_APPS_WITNESS_SERVICE_H_
