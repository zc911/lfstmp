/*============================================================================
 * File Name   : witness_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <fstream>
#include <boost/algorithm/string/split.hpp>
#include <glog/logging.h>
#include <sys/time.h>

#include "witness_service.h"
#include "image_service.h"
#include "config/config_val.h"

namespace dg {

WitnessAppsService::WitnessAppsService(const Config *config)
        : config_(config),
          engine_(*config),
          id_(0) {

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

    init();
}

WitnessAppsService::~WitnessAppsService() {

}

void WitnessAppsService::init(void) {
    string vModelFile  = (string) config_->Value(ConfigValue::VEHICLE_MODEL_MAPPING_FILE);
    string vColorFile  = (string) config_->Value(ConfigValue::VEHICLE_COLOR_MAPPING_FILE);
    string vSymbolFile = (string) config_->Value(ConfigValue::VEHICLE_SYMBOL_MAPPING_FILE);
    string pColorFile  = (string) config_->Value(ConfigValue::VEHICLE_PLATE_COLOR_MAPPING_FILE);
    string pTypeFile   = (string) config_->Value(ConfigValue::VEHICLE_PLATE_TYPE_MAPPING_FILE);
 
    init_vehicle_map(vModelFile, ",", vehicle_repo_);
    init_string_map(vColorFile, "=", color_repo_);
    init_string_map(vSymbolFile, "=", symbol_repo_); 
    init_string_map(pColorFile, "=", plate_color_repo_);
    init_string_map(pTypeFile, "=", plate_type_repo_);
}

int WitnessAppsService::parseInt(string str)
{
    return std::stoi(trimString(str), nullptr, 10);
}

string WitnessAppsService::trimString(string str)
{
    str.erase(0, str.find_first_not_of(" \n\r\t")); //prefixing spaces
    str.erase(str.find_last_not_of(" \n\r\t")+1);   //surfixing spaces
    return str;
}

void WitnessAppsService::init_string_map(string filename, string sep, vector<string>& array) {
    ifstream input(filename);

    int max = 0;
    vector<std::pair<int, string>> pairs;
    for(string line; std::getline(input, line); )
    {
        vector<string> tokens;
        boost::iter_split(tokens, line, boost::first_finder(sep));
        assert(tokens.size() == 2);

        int index = parseInt(tokens[0]);
        if (index > max) max = index;

        pairs.push_back(std::pair<int, string>(index, trimString(tokens[1])));
    }

    array.resize(max + 1);
    for(int i = 0; i <= max; i ++)
    {
        array[i] = unknown_string_;
    }

    for(const std::pair<int, string>& p : pairs)
    {
        array[p.first] = p.second;
    }
}

void WitnessAppsService::init_vehicle_map(string filename, string sep, vector<VehicleModel>& array) {
    ifstream input(filename);

    int max = 0;
    vector<std::pair<int, VehicleModel>> pairs;
    for(string line; std::getline(input, line); )
    {
        vector<string> tokens;
        boost::iter_split(tokens, line, boost::first_finder(sep));
        assert(tokens.size() == 10);

        int index = parseInt(tokens[0]);
        if (index > max) max = index;

        VehicleModel m;
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

        pairs.push_back(std::pair<int, VehicleModel>(index, m));
    }

    array.resize(max + 1);
    for(int i = 0; i <= max; i ++)
    {
        array[i].CopyFrom(unknown_vehicle_);
    }

    for(const std::pair<int, VehicleModel>& p : pairs)
    {
        array[p.first].CopyFrom(p.second);
    }
}

const string& WitnessAppsService::lookup_string(const vector<string>& array, int index)
{
    if (index < 0 || index > array.size())
    {
        return unknown_string_;
    }

    return array[index];
}

const VehicleModel& WitnessAppsService::lookup_vehicle(const vector<VehicleModel>& array, int index)
{
    if (index < 0 || index > array.size())
    {
        return unknown_vehicle_;
    }

    return array[index];
}

Operation WitnessAppsService::getOperation(const WitnessRequestContext& ctx) {
    Operation op;
    for (int i = 0; i < ctx.functions_size(); i++) {
        switch (ctx.functions(i)) {
            case RECFUNC_NONE:
                op.Set(OPERATION_NONE);
                break;
            case RECFUNC_VEHICLE:
                op.Set(OPERATION_VEHICLE);
                break;
            case RECFUNC_VEHICLE_DETECT:
                op.Set(OPERATION_VEHICLE_DETECT);
                break;
            case RECFUNC_VEHICLE_TRACK:
                op.Set(OPERATION_VEHICLE_TRACK);
                break;
            case RECFUNC_VEHICLE_STYLE:
                op.Set(OPERATION_VEHICLE_STYLE);
                break;
            case RECFUNC_VEHICLE_COLOR:
                op.Set(OPERATION_VEHICLE_COLOR);
                break;
            case RECFUNC_VEHICLE_MARKER:
                op.Set(OPERATION_VEHICLE_MARKER);
                break;
            case RECFUNC_VEHICLE_PLATE:
                op.Set(OPERATION_VEHICLE_PLATE);
                break;
            case RECFUNC_VEHICLE_FEATURE_VECTOR:
                op.Set(OPERATION_VEHICLE_FEATURE_VECTOR);
                break;
            case RECFUNC_FACE:
                op.Set(OPERATION_FACE);
                break;
            case RECFUNC_FACE_DETECTOR:
                op.Set(OPERATION_FACE_DETECTOR);
                break;
            case RECFUNC_FACE_FEATURE_VECTOR:
                op.Set(OPERATION_FACE_FEATURE_VECTOR);
                break;
            default:
                break;
        }
    }

    return op;
}

void WitnessAppsService::copyCutboard(const Box &b, Cutboard *cb) {
    cb->set_x(b.x);
    cb->set_y(b.y);
    cb->set_width(b.width);
    cb->set_height(b.height);
}

MatrixError WitnessAppsService::fillModel(Identification id,
                                          VehicleModel *model) {
    MatrixError err;
    const VehicleModel& m = lookup_vehicle(vehicle_repo_, id);
    model->CopyFrom(m);
    return err;
}

MatrixError WitnessAppsService::fillColor(const Vehicle::Color &color,
                                          Color *rcolor) {
    MatrixError err;

    rcolor->set_id(color.class_id);
    rcolor->set_colorname(lookup_string(color_repo_, color.class_id));
    rcolor->set_confidence(color.confidence);

    return err;
}

MatrixError WitnessAppsService::fillPlate(const Vehicle::Plate &plate,
                                          LicensePlate *rplate) {
    MatrixError err;
    rplate->set_platenum(plate.plate_num);
    copyCutboard(plate.box, rplate->mutable_cutboard());
    rplate->set_colorid(plate.color_id);
    rplate->set_color(lookup_string(plate_color_repo_, plate.color_id));
    rplate->set_typeid_(plate.plate_type);
    rplate->set_type(lookup_string(plate_type_repo_, plate.plate_type));
    rplate->set_confidence(plate.confidence);

    return err;
}

MatrixError WitnessAppsService::fillSymbols(const vector<Object*>& objects,
                                            RecognizedVehicle *vrec) {
    MatrixError err;

    int isize = symbol_repo_.size();
    int* indexes = new int[isize];
    for (int i = 0; i < isize; i++) indexes[i] = -1;
    for (const Object *object : objects) {
        LOG(INFO)<< "recognized marker: " << object->id() << ", type: " << object->type();
        if (object->type() != OBJECT_MARKER)
        {
            LOG(WARNING) << "unknown marker type: " << object->type();
            continue;
        }

        Marker *m = (Marker *)object;
        Identification mid = m->class_id();
        if (mid >= 0 && mid < isize)
        {
            SymbolItem *item = NULL;
            if (indexes[mid] < 0)
            {
                indexes[mid] = vrec->symbolitems_size();
                item = vrec->add_symbolitems();
                item->set_symbolid(mid);
                item->set_symbolname(lookup_string(symbol_repo_, mid));
            }
            else
            {
                item = vrec->mutable_symbolitems( indexes[mid] );
            }

            Symbol *s = item->add_symbols();
            s->set_confidence(m->detection().confidence);
            copyCutboard(m->detection().box, s->mutable_cutboard());
        }
    }
    delete indexes;

    return err;
}

MatrixError WitnessAppsService::getRecognizedVehicle(const Vehicle *vobj, RecognizedVehicle *vrec)
{
    MatrixError err;
    vrec->set_features(vobj->feature().Serialize());

    const Detection &d = vobj->detection();
    LOG(INFO) << "detected object: " << vobj->class_id();
    copyCutboard(d.box, vrec->mutable_cutboard());

    err = fillModel(vobj->class_id(), vrec->mutable_model());
    vrec->mutable_model()->set_confidence(vobj->confidence());
    if (err.code() < 0)
        return err;

    err = fillColor(vobj->color(), vrec->mutable_color());
    if (err.code() < 0)
        return err;

    err = fillPlate(vobj->plate(), vrec->mutable_licenseplate());
    if (err.code() < 0)
        return err;

    err = fillSymbols(vobj->children(), vrec);
    if (err.code() < 0)
        return err;

    return err;
}

MatrixError WitnessAppsService::getRecognizedFace(const Face *fobj, RecognizedFace *frec)
{
    MatrixError err;
    frec->set_confidence((float) fobj->confidence());
    frec->set_features(fobj->feature().Serialize());

    const Detection &d = fobj->detection();
    LOG(INFO)<< "detection id: " << d.id << ", deleted? " << d.deleted;
    copyCutboard(d.box, frec->mutable_cutboard());
    return err;
}

MatrixError WitnessAppsService::getRecognizeResult(Frame *frame,
                                                   WitnessResult *result) {
    MatrixError err;

    for (const Object *object : frame->objects()) {
        LOG(INFO)<< "recognized object: " << object->id() << ", type: " << object->type();
        switch(object->type())
        {
            case OBJECT_CAR:
            err = getRecognizedVehicle((Vehicle *)object, result->add_vehicles());
            break;

            case OBJECT_FACE:
            err = getRecognizedFace((Face *)object, result->add_faces());
            break;

            default:
            LOG(WARNING) << "unknown object type: " << object->type();
        }

        if (err.code() < 0)
        {
            break;
        }
    }

    return err;
}

bool WitnessAppsService::Recognize(const WitnessRequest *request,
                                   WitnessResponse *response) {
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);

    const string& sessionid = request->context().sessionid();

    if (!request->has_image() || !request->image().has_data()) {
        LOG(ERROR)<< "image descriptor does not exist";
        return false;
    }

    LOG(INFO)<< "Get Recognize request: " << sessionid
    << ", Image URI:" << request->image().data().uri();
    LOG(INFO)<< "Start processing: " << sessionid << "...";

    Mat image;
    MatrixError err = ImageService::ParseImage(request->image().data(), image);
    if (err.code() != 0) {
        LOG(ERROR)<< "parse image failed, " << err.message();
        return false;
    }

    Identification curr_id = id_++;  //TODO: make thread safe
    Frame *frame = new Frame(curr_id, image);
    frame->set_operation(getOperation(request->context()));

    FrameBatch framebatch(curr_id * 10, 1);
    framebatch.add_frame(frame);
    engine_.Process(&framebatch);

    //fill response
    WitnessResponseContext* ctx = response->mutable_context();
    ctx->set_sessionid(sessionid);
    ctx->mutable_requestts()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_requestts()->set_nanosecs((int64_t) curr_time.tv_usec);
    ctx->set_status("200");
    ctx->set_message("SUCCESS");

    //debug information of this request
    ::google::protobuf::Map<::std::string, ::dg::Time>& debugTs = *ctx
            ->mutable_debugts();

    WitnessResult *result = response->mutable_result();
    err = getRecognizeResult(frame, result);
    if (err.code() != 0) {
        LOG(ERROR)<< "get result from frame failed, " << err.message();
        return false;
    }

    WitnessImage *ret_image = result->mutable_image();
    ret_image->CopyFrom(request->image());
    ret_image->mutable_data()->set_width(image.cols);
    ret_image->mutable_data()->set_height(image.rows);
    //if ReturnsImage, compress image into data.bindata

    gettimeofday(&curr_time, NULL);
    ctx->mutable_responsets()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_responsets()->set_nanosecs((int64_t) curr_time.tv_usec);

    LOG(INFO) << "recognized objects: " << frame->objects().size() << endl;
    LOG(INFO) << "Finish processing: " << sessionid << "..." << endl;
    LOG(INFO) << "=======" << endl;
    return true;
}

bool WitnessAppsService::BatchRecognize(const WitnessBatchRequest *request,
                                        WitnessBatchResponse *response) {

    return true;
}

}
