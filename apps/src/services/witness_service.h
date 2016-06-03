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
#include "model/witness.grpc.pb.h"
#include "engine_service.h"
namespace dg {
using namespace ::dg::model;



class WitnessAppsService : public EngineService {
public:
    WitnessAppsService(const Config *config, string name);
    virtual ~WitnessAppsService();

    MatrixError Recognize(const WitnessRequest *request, WitnessResponse *response);

    MatrixError BatchRecognize(const WitnessBatchRequest *request,
                               WitnessBatchResponse *response);
private:
    string name_;
    const Config *config_;
    WitnessEngine engine_;
    Identification id_;

    //repo list
    string unknown_string_;
    VehicleModel unknown_vehicle_;
    vector<VehicleModel> vehicle_repo_;
    vector<string> vehicle_type_repo_;
    vector<string> color_repo_;
    vector<string> symbol_repo_;
    vector<string> plate_color_repo_;
    vector<string> plate_type_repo_;

    // library caffe is not thread safe(even crash some times) which means
    // only one frame/frameBatch could be processed at one time.
    // So we use a lock to keep the processing thread safe.
    std::mutex rec_lock_;

    void init(void);
    void init_string_map(string filename, string sep, vector<string> &array);
    void init_vehicle_map(string filename, string sep,
                          vector<VehicleModel> &array);
    const string &lookup_string(const vector<string> &array, int index);
    const VehicleModel &lookup_vehicle(const vector<VehicleModel> &array,
                                       int index);

    static string trimString(string str);
    static int parseInt(string str);
    static Operation getOperation(const WitnessRequestContext &ctx);
    static void copyCutboard(const Detection &d, Cutboard *cb);


    MatrixError checkRequest(const WitnessRequest &request);
    MatrixError checkRequest(const WitnessBatchRequest &requests);
    MatrixError checkWitnessImage(const WitnessImage &wImage);
    MatrixError fillModel(const Vehicle &vobj, RecognizedVehicle *vrec);
    MatrixError fillColor(const Vehicle::Color &color, Color *rcolor);
    MatrixError fillPlate(const Vehicle::Plate &plate, LicensePlate *rplate);
    MatrixError fillSymbols(const vector<Object *> &objects,
                            RecognizedVehicle *vrec);
    MatrixError getRecognizedVehicle(const Vehicle *vobj,
                                     RecognizedVehicle *vrec);
    MatrixError getRecognizedFace(const Face *fobj, RecognizedFace *frec);
    MatrixError getRecognizeResult(Frame *frame, WitnessResult *result);
    MatrixError getRecognizedPedestrain(const Pedestrain *pedestrain, RecognizedPedestrain *result);


};

}

#endif //MATRIX_APPS_WITNESS_SERVICE_H_
