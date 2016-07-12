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
#include "string_util.h"
#include "../config/config_val.h"
#include "grpc/spring_grpc.h"
#include "engine_service.h"
#include "witness_bucket.h"
//
namespace dg {
using namespace ::dg::model;


class WitnessAppsService: public EngineService {
public:
    WitnessAppsService(const Config *config, string name, int baseId = 0);
    virtual ~WitnessAppsService();

    MatrixError Recognize(const WitnessRequest *request, WitnessResponse *response);

    MatrixError BatchRecognize(const WitnessBatchRequest *request,
                               WitnessBatchResponse *response);
    MatrixError Index(const IndexRequest *request,
                      IndexResponse *response);
    MatrixError IndexTxt(const IndexTxtRequest *request,
                         IndexTxtResponse *response);
    string name_;
private:
    static void filterPlateType(string color,string plateNum,int &type){
        if(plateNum.size()<2)
            return;
        char first[2];
        memcpy(first,plateNum.c_str(),sizeof(first));
        if(color=="蓝"){
            type=1;
        }else if(color=="黄"){
            if(type==0){
                type=3;
            }else if(type==1){
                type=4;
            }
            if(first=="学"){

            }
        }else if(color=="黑"){
            if(first=="使"){
                type=10;
            }else if(first=="港"){
                type=11;
            }else{
                type=2;
            }
        }else if(color=="绿"){
            type=12;
        }else if(color=="白"){
            if(first=="WJ"){
                type=6;
            }else{
                type=5;
            }
        }
    }

    const Config *config_;
    WitnessEngine engine_;
    Identification id_;
    Identification base_id_;

    //repo list
    string unknown_string_;
    VehicleModelType unknown_vehicle_;
    vector<VehicleModelType> vehicle_repo_;
    vector<string> vehicle_type_repo_;
    vector<string> color_repo_;
    vector<string> symbol_repo_;
    vector<string> plate_color_repo_;
    vector<string> plate_type_repo_;
    vector<string> plate_color_gpu_repo_;
    string model_mapping_data_;
    string color_mapping_data_;
    string symbol_mapping_data_;
    string plate_color_mapping_data_;
    string plate_type_mapping_data_;
    string vehicle_type_mapping_data_;
    string plate_color_gpu_mapping_data_;
    vector<string> pedestrian_attr_type_repo_;
    string pedestrian_attr_mapping_data_;


    // library caffe is not thread safe(even crash some times) which means
    // only one frame/frameBatch could be processed at one time.
    // So we use a lock to keep the processing thread safe.
    std::mutex rec_lock_;

    void init(void);
    void init_string_map(string filename, string sep, vector<string> &array);
    void init_vehicle_map(string filename, string sep,
                          vector<VehicleModelType> &array);
    const string &lookup_string(const vector<string> &array, int index);
    const VehicleModelType &lookup_vehicle(const vector<VehicleModelType> &array,
                                           int index);

    static string trimString(string str);
    static int parseInt(string str);
    static Operation getOperation(const WitnessRequestContext &ctx);
    static void copyCutboard(const Detection &d, Cutboard *cb);


    MatrixError checkRequest(const WitnessRequest &request);
    MatrixError checkRequest(const WitnessBatchRequest &requests);
    MatrixError checkWitnessImage(const WitnessImage &wImage);
    MatrixError fillModel(const Vehicle &vobj, RecVehicle *vrec);
    MatrixError fillColor(const Vehicle::Color &color, Color *rcolor);
    MatrixError fillPlates(const vector<Vehicle::Plate> &plate, RecVehicle *vrec);
    MatrixError fillSymbols(const vector<Object *> &objects,
                            RecVehicle *vrec);
    MatrixError getRecognizedVehicle(const Vehicle *vobj,
                                     RecVehicle *vrec);
    MatrixError getRecognizedPedestrian(const Pedestrian *pobj,
                                        RecVehicle *vrec);
    MatrixError getRecognizedFace(const Face *fobj, RecFace *frec);
    MatrixError getRecognizeResult(Frame *frame, WitnessResult *result);
//    void getStorageData(Frame *frame,shared_ptr<VehicleObj> &result);

};

}

#endif //MATRIX_APPS_WITNESS_SERVICE_H_
