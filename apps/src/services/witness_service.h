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
namespace dg {
using namespace ::dg::model;



class WitnessAppsService : public EngineService {
public:
    WitnessAppsService(const Config *config, string name);
    virtual ~WitnessAppsService();

    MatrixError Recognize(const WitnessRequest *request, WitnessResponse *response);

    MatrixError BatchRecognize(const WitnessBatchRequest *request,
                               WitnessBatchResponse *response);
    MatrixError Index(const IndexRequest *request,
                           IndexResponse *response);
    string name_;
private:

  /*  static void readMappingFile(std::string filename,char c,map<int,string>&collect){

        FILE *fp = fopen(filename.c_str(),"r");
        char msg[1000];
        while(fgets(msg,sizeof(msg),fp)!=NULL){
            splitForMap(collect,msg,'=');
        }
    }
    void readCarModelFile(std::string filename){
        FILE *fp = fopen(filename.c_str(),"r");
        char msg[1000];
        while(fgets(msg,sizeof(msg),fp)!=NULL){
            vector<string> line;
            split(msg,line,',');
            if(line.size()!=10)
                continue;
            car_main_brand_collect_.insert(pair<int,string>(atoi(line[0]),line[5]));
            car_sub_brand_collect_.insert(pair<int,string>(atoi(line[0]),line[7]));
            year_model_collect_.insert(pair<int,string>(atoi(line[0]),line[9]));
            if(line[3]==0){
                car_head_tail_collect_.insert(pair<int,string>(atoi(line[0]),"head"));
            }else{
                car_head_tail_collect_.insert(pair<int,string>(atoi(line[0]),"end"));
            }
            car_style_collect_.insert(pair<int,string>(atoi(line[0]),line[2]));
        }

    }*/
    const Config *config_;
    WitnessEngine engine_;
    Identification id_;

    //repo list
    string unknown_string_;
    VehicleModelType unknown_vehicle_;
    vector<VehicleModelType> vehicle_repo_;
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
    MatrixError fillPlate(const Vehicle::Plate &plate, LicensePlate *rplate);
    MatrixError fillSymbols(const vector<Object *> &objects,
                            RecVehicle *vrec);
    MatrixError getRecognizedVehicle(const Vehicle *vobj,
                                     RecVehicle *vrec);
    MatrixError getRecognizedFace(const Face *fobj, RecFace *frec);
    MatrixError getRecognizeResult(Frame *frame, WitnessResult *result);
 //   MatrixError getRecognizedPedestrain(const Pedestrain *pedestrain, RecPedestrian *result);


};

}

#endif //MATRIX_APPS_WITNESS_SERVICE_H_
