//
// Created by chenzhen on 7/18/16.
//
#include "repo_service.h"

namespace dg {

bool RepoService::is_init_ = false;

RepoService::RepoService() {
    unknown_string_ = "UNKNOWN";
    unknown_vehicle_.set_typeid_(-1);
    unknown_vehicle_.set_type("UNKNOWN");
    unknown_vehicle_.set_ishead(-1);
    unknown_vehicle_.set_brandid(-1);
    unknown_vehicle_.set_brand("UNKNOWN");
    unknown_vehicle_.set_subbrandid(-1);
    unknown_vehicle_.set_subbrand("UNKNOWN");
    unknown_vehicle_.set_modelyearid(-1);
    unknown_vehicle_.set_modelyear("UNKNOWN");
    unknown_vehicle_.set_confidence(-1.0);
    is_gpu_plate_ = false;
}

void RepoService::Init(const Config &config) {
    if (!is_init_) {

        string vModelFile = (string) config.Value(VEHICLE_MODEL_MAPPING_FILE);
        string vColorFile = (string) config.Value(VEHICLE_COLOR_MAPPING_FILE);
        string vSymbolFile = (string) config.Value(VEHICLE_SYMBOL_MAPPING_FILE);
        string pColorFile = (string) config.Value(VEHICLE_PLATE_COLOR_MAPPING_FILE);
        string pColorGpuFile = (string) config.Value(RENDER_VEHICLE_PLATE_GPU_COLOR);
        string pTypeFile = (string) config.Value(VEHICLE_PLATE_TYPE_MAPPING_FILE);
        string pVtypeFile = (string) config.Value(VEHICLE_TYPE_MAPPING_FILE);
        string pPtypeFile = (string) config.Value(VEHICLE_PEDESTRIAN_ATTR_TYPE);
        init_vehicle_map(vModelFile, ",", vehicle_repo_);
        init_string_map(vColorFile, "=", color_repo_);
        init_string_map(vSymbolFile, "=", symbol_repo_);
        init_string_map(pColorFile, "=", plate_color_repo_);
        init_string_map(pColorGpuFile, "=", plate_color_gpu_repo_);
        init_string_map(pTypeFile, "=", plate_type_repo_);
        init_string_map(pVtypeFile, "=", vehicle_type_repo_);
        init_string_map(pPtypeFile, "=", pedestrian_attr_type_repo_);
        model_mapping_data_ = ReadStringFromFile(vModelFile, "r");
        color_mapping_data_ = ReadStringFromFile(vColorFile, "r");
        symbol_mapping_data_ = ReadStringFromFile(vSymbolFile, "r");
        plate_color_mapping_data_ = ReadStringFromFile(pColorFile, "r");
        plate_type_mapping_data_ = ReadStringFromFile(pTypeFile, "r");
        vehicle_type_mapping_data_ = ReadStringFromFile(pVtypeFile, "r");
        plate_color_gpu_mapping_data_ = ReadStringFromFile(pColorFile, "r");
        pedestrian_attr_mapping_data_ = ReadStringFromFile(pPtypeFile, "r");
        is_gpu_plate_ = (bool) config.Value(IS_GPU_PLATE);

        is_init_ = true;
    }
}

MatrixError RepoService::IndexTxt(const IndexTxtRequest *request,
                                  IndexTxtResponse *response) {
    MatrixError err;
    string data;
    switch (request->indextype()) {
        case INDEX_CAR_TYPE:
            data = model_mapping_data_;
            break;
        case INDEX_CAR_PLATE_COLOR:
            if (!is_gpu_plate_) {
                data = plate_color_mapping_data_;
            } else {
                data = plate_color_gpu_mapping_data_;
            }
            break;
        case INDEX_CAR_PLATE_TYPE:
            data = plate_type_mapping_data_;
            break;
        case INDEX_CAR_COLOR:
            data = color_mapping_data_;
            break;
        case INDEX_CAR_MARKER:
            data = symbol_mapping_data_;
            break;
        case INDEX_CAR_PEDESTRIAN_ATTR_TYPE:
            data = pedestrian_attr_mapping_data_;
            break;
    }
    response->set_context(data);


    return err;
}
void RepoService::init_string_map(string filename, string sep,
                                  vector<string> &array) {
    ifstream input(filename);

    int max = 0;
    vector<std::pair<int, string>> pairs;
    for (string line; std::getline(input, line);) {
        vector<string> tokens;
        boost::iter_split(tokens, line, boost::first_finder(sep));
        assert(tokens.size() == 2);

        int index = parseInt(tokens[0]);
        if (index > max)
            max = index;

        pairs.push_back(std::pair<int, string>(index, trimString(tokens[1])));
    }

    array.resize(max + 1);
    for (int i = 0; i <= max; i++) {
        array[i] = unknown_string_;
    }

    for (const std::pair<int, string> &p : pairs) {
        array[p.first] = p.second;
    }
}

void RepoService::init_vehicle_map(string filename, string sep,
                                   vector<VehicleModelType> &array) {
    ifstream input(filename);

    int max = 0;
    vector<std::pair<int, VehicleModelType>> pairs;
    map<int, string> typeMap;
    for (string line; std::getline(input, line);) {
        vector<string> tokens;
        boost::iter_split(tokens, line, boost::first_finder(sep));
        assert(tokens.size() == 10);

        int index = parseInt(tokens[0]);
        if (index > max)
            max = index;

        VehicleModelType m;
        m.set_typeid_(parseInt(tokens[1]));
        m.set_type(trimString(tokens[2]));
        m.set_ishead(parseInt(tokens[3]));
        m.set_brandid(parseInt(tokens[4]));
        m.set_brand(trimString(tokens[5]));
        m.set_subbrandid(parseInt(tokens[6]));
        m.set_subbrand(trimString(tokens[7]));
        m.set_modelyearid(parseInt(tokens[8]));
        m.set_modelyear(trimString(tokens[9]));
        m.set_confidence(-1.0);
        typeMap[m.typeid_()] = m.type();

        pairs.push_back(std::pair<int, VehicleModelType>(index, m));
    }

    array.resize(max + 1);
    for (int i = 0; i <= max; i++) {
        array[i].CopyFrom(unknown_vehicle_);
    }

    for (const std::pair<int, VehicleModelType> &p : pairs) {
        array[p.first].CopyFrom(p.second);
    }

    car_type_repo_.resize(typeMap.size() + 1);
    for (auto v : typeMap) {
        car_type_repo_[v.first] = v.second;
    }
}

const string &RepoService::lookup_string(const vector<string> &array,
                                         int index) {
    if (index < 0 || index > array.size()) {
        return unknown_string_;
    }

    return array[index];
}

const VehicleModelType &RepoService::lookup_vehicle(
    const vector<VehicleModelType> &array, int index) {
    if (index < 0 || index > array.size()) {
        return unknown_vehicle_;
    }

    return array[index];
}

MatrixError RepoService::FillModel(const Vehicle &vobj,
                                   RecVehicle *vrec) {
    MatrixError err;
    string type = lookup_string(vehicle_type_repo_, vobj.type());
    vrec->set_vehicletypename(type);

    if (vobj.type() == OBJECT_CAR) {
        const VehicleModelType &m = lookup_vehicle(vehicle_repo_, vobj.class_id());
        VehicleModelType *model = vrec->mutable_modeltype();
        model->CopyFrom(m);
        vrec->set_vehicletype(OBJ_TYPE_CAR);
    } else if (vobj.type() == OBJECT_BICYCLE) {
        vrec->set_vehicletype(OBJ_TYPE_BICYCLE);
    } else if (vobj.type() == OBJECT_TRICYCLE) {
        vrec->set_vehicletype(OBJ_TYPE_TRICYCLE);
    }
    return err;
}

MatrixError RepoService::FillColor(const Vehicle::Color &color,
                                   Color *rcolor) {
    MatrixError err;
    rcolor->set_colorid(color.class_id);
    rcolor->set_colorname(lookup_string(color_repo_, color.class_id));
    rcolor->set_confidence(color.confidence);

    return err;
}


MatrixError RepoService::FillPlates(const vector<Vehicle::Plate> &plates,
                                    RecVehicle *vrec) {
    MatrixError err;
    for (auto plate:plates) {
        LicensePlate *rplate = vrec->add_plates();
        rplate->set_platetext(plate.plate_num);
        Detection d;
        d.box = plate.box;
        CopyCutboard(d, rplate->mutable_cutboard());
        rplate->mutable_color()->set_colorid(plate.color_id);
        int typeId = plate.plate_type;
        if (is_gpu_plate_) {
            rplate->mutable_color()->set_colorname(lookup_string(plate_color_gpu_repo_, plate.color_id));
            filterPlateType(rplate->color().colorname(), plate.plate_num, typeId);
        } else {

            rplate->mutable_color()->set_colorname(lookup_string(plate_color_repo_, plate.color_id));
        }
        rplate->mutable_color()->set_confidence(plate.confidence);
        rplate->set_typeid_(typeId);
        rplate->set_typename_(lookup_string(plate_type_repo_, typeId));
        rplate->set_confidence(plate.confidence);
        vrec->mutable_plate()->CopyFrom(*rplate);
    }

    return err;
}
MatrixError RepoService::FillSymbols(const vector<Object *> &objects,
                                     RecVehicle *vrec) {
    MatrixError err;

    int isize = symbol_repo_.size();
    int *indexes = new int[isize];
    for (int i = 0; i < isize; i++)
        indexes[i] = -1;
    for (const Object *object : objects) {

        if (object->type() != OBJECT_MARKER) {
            LOG(WARNING) << "unknown marker type: " << object->type();
            continue;
        }

        Marker *m = (Marker *) object;
        Identification mid = m->class_id();
        if (mid >= 0 && mid < isize) {
            VehicleSymbol *item = NULL;
            if (indexes[mid] < 0) {
                indexes[mid] = vrec->symbols_size();
                item = vrec->add_symbols();
                item->set_symbolid(mid);
                item->set_symbolname(lookup_string(symbol_repo_, mid));
            }
            else {
                item = vrec->mutable_symbols(indexes[mid]);
            }

            Symbol *s = item->add_symbols();
            s->set_confidence(m->detection().confidence);
            CopyCutboard(m->detection(), s->mutable_cutboard());
        }
    }
    delete indexes;

    return err;
}

MatrixError RepoService::Index(const IndexRequest *request,
                               IndexResponse *response) {
    MatrixError err;
    switch (request->indextype()) {
        case INDEX_CAR_BRAND: {
            if (response->has_index())
                break;
            BrandIndex *sIndex = response->mutable_brandindex();

            for (int i = 0; i < vehicle_repo_.size(); i++) {
                VehicleModelType model = vehicle_repo_[i];
                BrandIndex_Item *item = sIndex->add_items();
                item->set_mainbrandid(model.brandid());
                item->set_subbrandid(model.subbrandid());
                item->set_yearmodelid(model.modelyearid());
                item->set_mainbrandname(model.brand());
                item->set_subbrandname(model.subbrand());
                item->set_yearmodelname(model.modelyear());
            }
            break;
        }
        case INDEX_CAR_TYPE: {
            if (response->has_brandindex())
                break;
            CommonIndex *cIndex = response->mutable_index();
            for (int i = 0; i < car_type_repo_.size(); i++) {
                string value = car_type_repo_[i].data();
                CommonIndex_Item *item = cIndex->add_items();
                item->set_id(i);
                item->set_name(value);
            }
            break;
        }
        case INDEX_CAR_PLATE_COLOR: {

            if (response->has_brandindex())
                break;

            CommonIndex *cIndex = response->mutable_index();
            if (!is_gpu_plate_) {
                for (int i = 0; i < plate_color_repo_.size(); i++) {
                    string value = plate_color_repo_[i].data();
                    CommonIndex_Item *item = cIndex->add_items();
                    item->set_id(i);
                    item->set_name(value);
                }
            } else {
                for (int i = 0; i < plate_color_gpu_repo_.size(); i++) {
                    string value = plate_color_gpu_repo_[i].data();
                    CommonIndex_Item *item = cIndex->add_items();
                    item->set_id(i);
                    item->set_name(value);
                }
            }
            break;
        }
        case INDEX_CAR_PLATE_TYPE: {
            if (response->has_brandindex())
                break;
            CommonIndex *cIndex = response->mutable_index();
            for (int i = 0; i < plate_type_repo_.size(); i++) {
                string value = plate_type_repo_[i].data();
                CommonIndex_Item *item = cIndex->add_items();
                item->set_id(i);
                item->set_name(value);
            }
            break;
        }
        case INDEX_CAR_COLOR: {
            if (response->has_brandindex())
                break;
            CommonIndex *cIndex = response->mutable_index();
            for (int i = 0; i < color_repo_.size(); i++) {
                string value = color_repo_[i].data();
                CommonIndex_Item *item = cIndex->add_items();
                item->set_id(i);
                item->set_name(value);
            }
            break;
        }
        case INDEX_CAR_MARKER: {
            if (response->has_brandindex())
                break;
            CommonIndex *cIndex = response->mutable_index();
            for (int i = 0; i < symbol_repo_.size(); i++) {
                string value = symbol_repo_[i].data();
                CommonIndex_Item *item = cIndex->add_items();
                item->set_id(i);
                item->set_name(value);
            }
            break;
        }
        case INDEX_CAR_PEDESTRIAN_ATTR_TYPE: {
            if (response->has_brandindex())
                break;
            CommonIndex *cIndex = response->mutable_index();
            for (int i = 0; i < pedestrian_attr_type_repo_.size(); i++) {
                string value = pedestrian_attr_type_repo_[i].data();
                CommonIndex_Item *item = cIndex->add_items();
                item->set_id(i);
                item->set_name(value);
            }
            break;
        }
    }

    return err;
}

void RepoService::filterPlateType(string color, string plateNum, int &type) {
    if (plateNum.size() < 2)
        return;
    char first[2];
    memcpy(first, plateNum.c_str(), sizeof(first));
    if (color == "蓝") {
        type = 1;
    } else if (color == "黄") {
        if (type == 0) {
            type = 3;
        } else if (type == 1) {
            type = 4;
        }
        if (first == "学") {

        }
    } else if (color == "黑") {
        if (first == "使") {
            type = 10;
        } else if (first == "港") {
            type = 11;
        } else {
            type = 2;
        }
    } else if (color == "绿") {
        type = 12;
    } else if (color == "白") {
        if (first == "WJ") {
            type = 6;
        } else {
            type = 5;
        }
    }
}

}