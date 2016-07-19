//
// Created by chenzhen on 7/18/16.
//

#ifndef PROJECT_REPO_SERVICE_H
#define PROJECT_REPO_SERVICE_H

#include <string>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include "config.h"
#include "string_util.h"
#include "../config/config_val.h"
#include "model/witness.grpc.pb.h"
#include "matrix_engine/model/model.h"

using namespace std;
using namespace ::dg::model;
namespace dg {

class RepoService {
public:
    static RepoService *GetInstance() {
        if (instance_ == NULL) {
            instance_ = new RepoService();
        }
        return instance_;
    }

    void Init(const Config &config);


    MatrixError Index(const IndexRequest *request,
                      IndexResponse *response);
    MatrixError IndexTxt(const IndexTxtRequest *request,
                         IndexTxtResponse *response);

    MatrixError FillModel(const Vehicle &vobj, RecVehicle *vrec);
    MatrixError FillColor(const Vehicle::Color &color, Color *rcolor);
    MatrixError FillPlates(const vector<Vehicle::Plate> &plate, RecVehicle *vrec);
    MatrixError FillSymbols(const vector<Object *> &objects,
                            RecVehicle *vrec);
    string FindVehicleTypeName(ObjectType type) {
        return lookup_string(vehicle_type_repo_, type);
    }
    string FindPedestrianAttrName(int attrId) {
        return lookup_string(pedestrian_attr_type_repo_, attrId);
    }

    static void CopyCutboard(const Detection &b, Cutboard *cb) {
        cb->set_x(b.box.x);
        cb->set_y(b.box.y);
        cb->set_width(b.box.width);
        cb->set_height(b.box.height);
        cb->set_confidence(b.confidence);
    }

private:
    RepoService();

    static RepoService *instance_;
    static bool is_init_;

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
    bool is_gpu_plate_;

    const string &lookup_string(const vector<string> &array, int index);
    const VehicleModelType &lookup_vehicle(const vector<VehicleModelType> &array,
                                           int index);
    void init_string_map(string filename, string sep, vector<string> &array);
    void init_vehicle_map(string filename, string sep,
                          vector<VehicleModelType> &array);

    int parseInt(string str) {
        return std::stoi(trimString(str), nullptr, 10);
    }

    string trimString(string str) {
        str.erase(0, str.find_first_not_of(" \n\r\t"));  //prefixing spaces
        str.erase(str.find_last_not_of(" \n\r\t") + 1);   //surfixing spaces
        return str;
    }

    void filterPlateType(string color, string plateNum, int &type);

};

}
#endif //PROJECT_REPO_SERVICE_H
