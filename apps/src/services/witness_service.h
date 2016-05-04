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

    //repo list
    string unknown_string_;
    VehicleModel unknown_vehicle_;
    vector<VehicleModel> vehicle_repo_;
    vector<string> color_repo_;
    vector<string> symbol_repo_;
    vector<string> plate_color_repo_;
    vector<string> plate_type_repo_;

    void init(void);
    void init_string_map(string filename, string sep, vector<string>& array);
    void init_vehicle_map(string filename, string sep, vector<VehicleModel>& array);
    const string& lookup_string(const vector<string>& array, int index);
    const VehicleModel& lookup_vehicle(const vector<VehicleModel>& array, int index);

    static string trimString(string str);
    static int parseInt(string str);
    static Operation getOperation(const WitnessRequestContext& ctx);
    static void copyCutboard(const Box &b, Cutboard *cb);

    MatrixError fillModel(Identification id, VehicleModel *model);
    MatrixError fillColor(const Vehicle::Color &color, Color *rcolor);
    MatrixError fillPlate(const Vehicle::Plate &plate, LicensePlate *rplate);
    MatrixError fillSymbols(const vector<Object*>& objects, RecognizedVehicle *vrec);
    MatrixError getRecognizedVehicle(const Vehicle *vobj, RecognizedVehicle *vrec);
    MatrixError getRecognizedFace(const Face *fobj, RecognizedFace *frec);
    MatrixError getRecognizeResult(Frame *frame, WitnessResult *result);
};

}

#endif //MATRIX_APPS_WITNESS_SERVICE_H_
