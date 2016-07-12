/*============================================================================
 * File Name   : witness_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description :
 * ==========================================================================*/

#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "pbjson/pbjson.hpp"
#include <boost/algorithm/string/split.hpp>
#include <glog/logging.h>
#include <sys/time.h>
#include <matrix_engine/model/model.h>
#include <google/protobuf/text_format.h>
#include "codec/base64.h"
#include "debug_util.h"
#include "witness_service.h"
#include "image_service.h"
#include "string_util.h"
#include "log/log_val.h"
//
using namespace std;
namespace dg {

static int SHIFT_COLOR = 1000;
WitnessAppsService::WitnessAppsService(const Config *config, string name)
    : config_(config),
      engine_(*config),
      id_(0) {
    name_ = name;
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
    string vModelFile = (string) config_->Value(VEHICLE_MODEL_MAPPING_FILE);
    string vColorFile = (string) config_->Value(VEHICLE_COLOR_MAPPING_FILE);
    string vSymbolFile = (string) config_->Value(VEHICLE_SYMBOL_MAPPING_FILE);
    string pColorFile = (string) config_->Value(VEHICLE_PLATE_COLOR_MAPPING_FILE);
    string pColorGpuFile = (string) config_->Value(RENDER_VEHICLE_PLATE_GPU_COLOR);
    string pTypeFile = (string) config_->Value(VEHICLE_PLATE_TYPE_MAPPING_FILE);
    string pVtypeFile = (string) config_->Value(VEHICLE_TYPE_MAPPING_FILE);
    string pPtypeFile = (string) config_->Value(VEHICLE_PEDESTRIAN_ATTR_TYPE);
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
    //advanced color
    int size = color_repo_.size();
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            string value = string(color_repo_[i]) + color_repo_[j];
            color_repo_.push_back(value);
        }
    }
}

int WitnessAppsService::parseInt(string str) {
    return std::stoi(trimString(str), nullptr, 10);
}

string WitnessAppsService::trimString(string str) {
    str.erase(0, str.find_first_not_of(" \n\r\t"));  //prefixing spaces
    str.erase(str.find_last_not_of(" \n\r\t") + 1);   //surfixing spaces
    return str;
}
MatrixError WitnessAppsService::IndexTxt(const IndexTxtRequest *request,
                                         IndexTxtResponse *response) {
    MatrixError err;
    string data;
    bool gpuplate = (bool) config_->Value(IS_GPU_PLATE);
    switch (request->indextype()) {
        case INDEX_CAR_TYPE:
            data = model_mapping_data_;
            break;
        case INDEX_CAR_MAIN_BRAND:
            data = model_mapping_data_;
            break;
        case INDEX_CAR_SUB_BRAND:
            data = model_mapping_data_;
            break;
        case INDEX_CAR_YEAR_MODEL:
            data = model_mapping_data_;
            break;
        case INDEX_CAR_PLATE_COLOR:
            if (!gpuplate) {
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
void WitnessAppsService::init_string_map(string filename, string sep,
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

void WitnessAppsService::init_vehicle_map(string filename, string sep,
                                          vector<VehicleModelType> &array) {
    ifstream input(filename);

    int max = 0;
    vector<std::pair<int, VehicleModelType>> pairs;
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

        pairs.push_back(std::pair<int, VehicleModelType>(index, m));
    }

    array.resize(max + 1);
    for (int i = 0; i <= max; i++) {
        array[i].CopyFrom(unknown_vehicle_);
    }

    for (const std::pair<int, VehicleModelType> &p : pairs) {
        array[p.first].CopyFrom(p.second);
    }
}

const string &WitnessAppsService::lookup_string(const vector<string> &array,
                                                int index) {
    if (index < 0 || index > array.size()) {
        return unknown_string_;
    }

    return array[index];
}

const VehicleModelType &WitnessAppsService::lookup_vehicle(
    const vector<VehicleModelType> &array, int index) {
    if (index < 0 || index > array.size()) {
        return unknown_vehicle_;
    }

    return array[index];
}

Operation WitnessAppsService::getOperation(const WitnessRequestContext &ctx) {
    Operation op;
    int type = ctx.type();
    for (int i = 0; i < ctx.functions_size(); i++) {
        switch (ctx.functions(i)) {
            case RECFUNC_NONE:
                op.Set(OPERATION_NONE);
                break;
            case RECFUNC_VEHICLE:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE);
                break;
            case RECFUNC_VEHICLE_DETECT:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_DETECT);
                break;
            case RECFUNC_VEHICLE_TRACK:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_TRACK);
                break;
            case RECFUNC_VEHICLE_STYLE:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_STYLE);
                break;
            case RECFUNC_VEHICLE_COLOR:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_COLOR);
                break;
            case RECFUNC_VEHICLE_MARKER:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_MARKER);
                break;
            case RECFUNC_VEHICLE_PLATE:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_PLATE);
                break;
            case RECFUNC_VEHICLE_FEATURE_VECTOR:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_FEATURE_VECTOR);
                break;
            case RECFUNC_VEHICLE_PEDESTRIAN_ATTR:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_PEDESTRIAN_ATTR);
                break;
            case RECFUNC_FACE:
                if ((type == REC_TYPE_FACE) || (type == REC_TYPE_ALL) || (type == REC_TYPE_DEFAULT))
                    op.Set(OPERATION_FACE);
                break;
            case RECFUNC_FACE_DETECTOR:
                if ((type == REC_TYPE_FACE) || (type == REC_TYPE_ALL) || (type == REC_TYPE_DEFAULT))
                    op.Set(OPERATION_FACE_DETECTOR);
                break;
            case RECFUNC_FACE_FEATURE_VECTOR:
                if ((type == REC_TYPE_FACE) || (type == REC_TYPE_ALL) || (type == REC_TYPE_DEFAULT))
                    op.Set(OPERATION_FACE_FEATURE_VECTOR);
                break;
            default:
                break;
        }
    }

    return op;
}

void WitnessAppsService::copyCutboard(const Detection &b, Cutboard *cb) {
    cb->set_x(b.box.x);
    cb->set_y(b.box.y);
    cb->set_width(b.box.width);
    cb->set_height(b.box.height);
    cb->set_confidence(b.confidence);
}

MatrixError WitnessAppsService::fillModel(const Vehicle &vobj,
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

MatrixError WitnessAppsService::fillColor(const Vehicle::Color &color,
                                          Color *rcolor) {
    MatrixError err;
    rcolor->set_colorid(color.class_id);
    rcolor->set_colorname(lookup_string(color_repo_, color.class_id));
    rcolor->set_confidence(color.confidence);

    return err;
}


MatrixError WitnessAppsService::fillPlates(const vector<Vehicle::Plate> &plates,
                                           RecVehicle *vrec) {
    bool gpuplate = (bool) config_->Value(IS_GPU_PLATE);
    MatrixError err;
    for (auto plate:plates) {
        LicensePlate *rplate = vrec->add_plates();
        rplate->set_platetext(plate.plate_num);
        Detection d;
        d.box = plate.box;
        copyCutboard(d, rplate->mutable_cutboard());
        rplate->mutable_color()->set_colorid(plate.color_id);
        int typeId=plate.plate_type;
        if (gpuplate) {
            rplate->mutable_color()->set_colorname(lookup_string(plate_color_gpu_repo_, plate.color_id));
            filterPlateType(rplate->color().colorname(),plate.plate_num,typeId);
        } else {

            rplate->mutable_color()->set_colorname(lookup_string(plate_color_repo_, plate.color_id));
        }
        rplate->mutable_color()->set_confidence(plate.confidence);
        rplate->set_typeid_(plate.plate_type);
        rplate->set_typename_(lookup_string(plate_type_repo_, plate.plate_type));
        rplate->set_confidence(plate.confidence);
        vrec->mutable_plate()->CopyFrom(*rplate);
    }

    return err;
}
MatrixError WitnessAppsService::fillSymbols(const vector<Object *> &objects,
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
            copyCutboard(m->detection(), s->mutable_cutboard());
        }
    }
    delete indexes;

    return err;
}

MatrixError WitnessAppsService::getRecognizedPedestrian(const Pedestrian *pobj,
                                                        RecVehicle *vrec) {
    MatrixError err;
    std::vector<Pedestrian::Attr> attrs = pobj->attrs();

    const Detection &d = pobj->detection();

    copyCutboard(d, vrec->mutable_img()->mutable_cutboard());
    vrec->set_vehicletype(OBJ_TYPE_PEDESTRIAN);
    string type = lookup_string(vehicle_type_repo_, pobj->type());
    vrec->set_vehicletypename(type);
    for (int i = 0; i < attrs.size(); i++) {
        PedestrianAttr *attr = vrec->add_pedestrianattrs();
        attr->set_attrid(attrs[i].index);
        attr->set_confidence(attrs[i].confidence);
        attr->set_attrname(lookup_string(pedestrian_attr_type_repo_, i));
    }

    return err;
}

MatrixError WitnessAppsService::getRecognizedVehicle(const Vehicle *vobj,
                                                     RecVehicle *vrec) {
    MatrixError err;
    vrec->set_features(vobj->feature().Serialize());


    const Detection &d = vobj->detection();

    copyCutboard(d, vrec->mutable_img()->mutable_cutboard());
    err = fillModel(*vobj, vrec);
    vrec->mutable_modeltype()->set_confidence(vobj->confidence());
    if (err.code() < 0)
        return err;

    err = fillColor(vobj->color(), vrec->mutable_color());
    if (err.code() < 0)
        return err;

    err = fillPlates(vobj->plates(), vrec);
    if (err.code() < 0)
        return err;

    err = fillSymbols(vobj->children(), vrec);
    if (err.code() < 0)
        return err;

    return err;
}

MatrixError WitnessAppsService::getRecognizedFace(const Face *fobj,
                                                  RecFace *frec) {
    MatrixError err;
    frec->set_confidence((float) fobj->confidence());
    frec->set_features(fobj->feature().Serialize());

    const Detection &d = fobj->detection();
    copyCutboard(d, frec->mutable_img()->mutable_cutboard());
    return err;
}
/*
MatrixError WitnessAppsService::getRecognizedPedestrain(
    const Pedestrain *pedestrain, RecPedestrian *result) {
    MatrixError err;
    const Detection &d = pedestrain->detection();
    result->set_confidence(d.confidence);
    copyCutboard(d, result->mutable_img()->mutable_cutboard());
    return err;
}
*/
MatrixError WitnessAppsService::getRecognizeResult(Frame *frame,
                                                   WitnessResult *result) {
    MatrixError err;

    for (const Object *object : frame->objects()) {
        DLOG(INFO) << "recognized object: " << object->id() << ", type: " << object->type();
        switch (object->type()) {
            case OBJECT_CAR:
            case OBJECT_BICYCLE:
            case OBJECT_TRICYCLE:
                err = getRecognizedVehicle((Vehicle *) object, result->add_vehicles());
                break;
            case OBJECT_PEDESTRIAN:
                err = getRecognizedPedestrian((Pedestrian *) object, result->add_vehicles());
                break;
            case OBJECT_FACE:
                err = getRecognizedFace((Face *) object, result->add_faces());
                break;
            default:
                LOG(WARNING) << "unknown object type: " << object->type();
                break;
        }

        if (err.code() < 0) {
            break;
        }
    }

    return err;
}

MatrixError WitnessAppsService::checkWitnessImage(const WitnessImage &wImage) {
    MatrixError err;
    if (wImage.data().uri().size() == 0
        && wImage.data().bindata().size() == 0) {
        LOG(ERROR) << "image uri and bindata are empty both";
        err.set_code(-1);
        err.set_message("image uri and bindata are empty both");
        return err;
    }
    if (wImage.data().uri().size() > 0) {
        string pre = findPrefix(wImage.data().uri(), ':');
        if (pre != "http" && pre != "https" && pre != "ftp" && pre != "file") {
            LOG(ERROR) << "Invalid URI: " << wImage.data().uri() << endl;
            err.set_code(-1);
            err.set_message("Invalid Image URI");
            return err;
        }
    }

    return err;
}

MatrixError WitnessAppsService::checkRequest(const WitnessRequest &request) {
    MatrixError err;
    if (!request.has_image() || !request.image().has_data()) {
        LOG(ERROR) << "image descriptor does not exist";
        err.set_code(-1);
        err.set_message("image descriptor does not exist");
        return err;
    }
    err = checkWitnessImage(request.image());
    if (err.code() != 0) {
        return err;
    }
    return err;
}

MatrixError WitnessAppsService::checkRequest(
    const WitnessBatchRequest &requests) {
    MatrixError err;
    int size = requests.images().size();
    if (size == 0) {
        err.set_code(-1);
        err.set_message("Not data within batch requests");
        return err;
    }

    for (int i = 0; i < size; ++i) {
        WitnessImage image = requests.images(i);
        err = checkWitnessImage(image);
        if (err.code() != 0) {
            return err;
        }
    }

    return err;
}
void storage(Frame *frame, VehicleObj *client_request_obj, string storageAddress
) {
    /*   */

}
MatrixError WitnessAppsService::Recognize(const WitnessRequest *request,
                                          WitnessResponse *response) {
    VLOG(VLOG_RUNTIME_DEBUG) << "Recognize using WitnessAppsService" << name_ << endl;
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    long long timestamp = curr_time.tv_sec * 1000 + curr_time.tv_usec / 1000;
    VLOG(VLOG_SERVICE) << "Received image timestamp: " << timestamp << endl;
    const string &sessionid = request->context().sessionid();
    MatrixError err = checkRequest(*request);
    if (err.code() != 0) {
        LOG(ERROR) << "Check request failed" << endl;
        return err;
    }

    VLOG(VLOG_SERVICE) << "Get Recognize request: " << sessionid
        << ", Image URI:" << request->image().data().uri();
    VLOG(VLOG_SERVICE) << "Start processing: " << sessionid << "...";

    struct timeval start, end;
    gettimeofday(&start, NULL);
    ROIImages roiimages;

    err = ImageService::ParseImage(request->image(), roiimages);
    if (err.code() != 0) {
        LOG(ERROR) << "parse image failed, " << err.message();
        return err;
    }
    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Parse Image cost: " << TimeCostInMs(start, end) << endl;

    Identification curr_id = id_++;  //TODO: make thread safe
    Frame *frame = new Frame(curr_id, roiimages.data);
    frame->set_operation(getOperation(request->context()));
    frame->set_roi(roiimages.rois);

    FrameBatch framebatch(curr_id * 10);
    framebatch.AddFrame(frame);
    gettimeofday(&start, NULL);
    //fill srcmetadata

    if (request->image().has_witnessmetadata() && request->image().witnessmetadata().timestamp() != 0) {
        timestamp = request->image().witnessmetadata().timestamp();
    }
    rec_lock_.lock();
    engine_.Process(&framebatch);
    rec_lock_.unlock();
    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Rec Image cost(pure): " << TimeCostInMs(start, end) << endl;

    gettimeofday(&start, NULL);

    //fill response
    WitnessResponseContext *ctx = response->mutable_context();
    ctx->set_sessionid(sessionid);
    ctx->mutable_requestts()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_requestts()->set_nanosecs((int64_t) curr_time.tv_usec);
    ctx->set_status("200");
    ctx->set_message("SUCCESS");

    //debug information of this request
    ::google::protobuf::Map<::std::string, ::dg::Time> &debugTs = *ctx
        ->mutable_debugts();
    WitnessResult *result = response->mutable_result();
    result->mutable_image()->mutable_data()->set_uri(
        request->image().data().uri());
    result->mutable_image()->mutable_data()->set_height(frame->payload()->data().rows);
    result->mutable_image()->mutable_data()->set_width(frame->payload()->data().cols);

    err = getRecognizeResult(frame, result);

    if (err.code() != 0) {
        LOG(ERROR) << "get result from frame failed, " << err.message();
        return err;
    }

    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Parse results cost: " << TimeCostInMs(start, end) << endl;

    // return back the image data
//    WitnessImage *ret_image = result->mutable_image();
//    ret_image->CopyFrom(request->image());
//    ret_image->mutable_data()->set_width(image.cols);
//    ret_image->mutable_data()->set_height(image.rows);
    //if ReturnsImage, compress image into data.bindata

    gettimeofday(&curr_time, NULL);
    ctx->mutable_responsets()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_responsets()->set_nanosecs((int64_t) curr_time.tv_usec);
    bool storageEnabled = (bool) config_->Value(STORAGE_ENABLED);
    if (storageEnabled) {
        string storageAddress;
        if (request->context().has_storage()) {
            storageAddress = (string) request->context().storage().address();
        } else {
            storageAddress = (string) config_->Value(STORAGE_ADDRESS);
        }
        int dbTypeInt = (int) config_->Value(STORAGE_DB_TYPE);
        DBType dbType;
        switch (dbTypeInt) {
            case 0:
                dbType = KAFKA;
                break;
            default:
                dbType = KAFKA;
                break;
        }
        const WitnessResult &r = response->result();
        if (r.vehicles_size() != 0) {
            unique_lock<mutex> lock(WitnessBucket::Instance().mt_push);
            shared_ptr<WitnessVehicleObj> client_request_obj(new WitnessVehicleObj);
            client_request_obj->mutable_storage()->set_address(storageAddress);
            for (int i = 0; i < r.vehicles_size(); i++) {
                Cutboard c = r.vehicles(i).img().cutboard();
                Mat roi(frame->payload()->data(), Rect(c.x(), c.y(), c.width(), c.height()));
                RecVehicle *v = client_request_obj->mutable_vehicleresult()->add_vehicle();
                v->CopyFrom(r.vehicles(i));
                bool enablecutboard = (bool) config_->Value("EnableCutboard");
                if (enablecutboard) {
                    vector<char> data(roi.datastart, roi.dataend);
                    string imgdata = Base64::Encode(data);
                    v->mutable_img()->mutable_img()->set_bindata(imgdata);
                }
            }
            VehicleObj *vehicleObj = client_request_obj->mutable_vehicleresult();
            //origin img info
            vehicleObj->mutable_img()->set_uri(request->image().data().uri());
            vehicleObj->mutable_img()->set_height(frame->payload()->data().rows);
            vehicleObj->mutable_img()->set_width(frame->payload()->data().cols);
            //src metadata
            vehicleObj->mutable_metadata()->CopyFrom(request->image().witnessmetadata());
            vehicleObj->mutable_metadata()->set_timestamp(timestamp);
            //      string s;
            //       google::protobuf::TextFormat::PrintToString(*client_request_obj.get(), &s);
            //        VLOG(VLOG_SERVICE) << s << endl;
            WitnessBucket::Instance().Push(client_request_obj);
            lock.unlock();
        }
    }

    VLOG(VLOG_SERVICE) << "recognized objects: " << frame->objects().size() << endl;
    VLOG(VLOG_SERVICE) << "Finish processing: " << sessionid << "..." << endl;
    VLOG(VLOG_SERVICE) << "=======" << endl;

    return err;

}

MatrixError WitnessAppsService::BatchRecognize(
    const WitnessBatchRequest *batchRequest,
    WitnessBatchResponse *batchResponse) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Batch recognize using " << name_ << endl;
    VLOG(VLOG_SERVICE) << "Batch recognize using " << name_ << endl;
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    MatrixError err;
    long long timestamp = curr_time.tv_sec * 1000 + curr_time.tv_usec / 1000;
    const string &sessionid = batchRequest->context().sessionid();

    const ::google::protobuf::RepeatedPtrField<::dg::model::WitnessImage> &images =
        batchRequest->images();

    VLOG(VLOG_SERVICE) << "Get Batch Recognize request: " << sessionid << ", batch size:" << images.size() << endl;
    VLOG(VLOG_SERVICE) << "Start processing: " << sessionid << "...";

    err = checkRequest(*batchRequest);
    if (err.code() != 0) {
        LOG(ERROR) << "Check request failed: " << sessionid << endl;
        return err;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    Identification curr_id = id_++;
    FrameBatch framebatch(curr_id * 10);
    vector<WitnessImage> imgDesc;
    vector<ROIImages> roiimages;
    vector<SrcMetadata> srcMetadatas;
    vector<
        ::google::protobuf::RepeatedPtrField<
            const ::dg::model::WitnessRelativeROI> > roisr;
    vector<
        ::google::protobuf::RepeatedPtrField<
            const ::dg::model::WitnessMarginROI> > roism;

    ::google::protobuf::RepeatedPtrField<const ::dg::model::WitnessImage>::iterator itr =
        images.begin();
    while (itr != images.end()) {
        if (itr->has_witnessmetadata() && itr->witnessmetadata().timestamp() != 0) {
            timestamp = itr->witnessmetadata().timestamp();
        }
        SrcMetadata metadata;
        metadata.CopyFrom(itr->witnessmetadata());
        metadata.set_timestamp(timestamp);
        srcMetadatas.push_back(metadata);
        imgDesc.push_back(const_cast<WitnessImage &>(*itr));
        itr++;
    }

    ImageService::ParseImage(imgDesc, roiimages, 10, true);

    for (int i = 0; i < roiimages.size(); ++i) {
        ROIImages image = roiimages[i];
        Identification curr_id = id_++;  //TODO: make thread safe
        Frame *frame = new Frame(curr_id, image.data);
        frame->set_operation(getOperation(batchRequest->context()));
        frame->set_roi(image.rois);

        framebatch.AddFrame(frame);
    }

    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Parse batch Image cost: " << TimeCostInMs(start, end) << endl;

    gettimeofday(&start, NULL);
    DLOG(INFO) << "Request batch size: " << framebatch.batch_size() << endl;
    rec_lock_.lock();
    engine_.Process(&framebatch);
    rec_lock_.unlock();
    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Rec batch Image cost(pure): " << TimeCostInMs(start, end) << endl;

    gettimeofday(&start, NULL);
    //fill response
    WitnessResponseContext *ctx = batchResponse->mutable_context();
    ctx->set_sessionid(sessionid);
    ctx->mutable_requestts()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_requestts()->set_nanosecs((int64_t) curr_time.tv_usec);
    ctx->set_status("200");
    ctx->set_message("SUCCESS");

    //debug information of this request
    ::google::protobuf::Map<::std::string, ::dg::Time> &debugTs = *ctx
        ->mutable_debugts();

    vector<Frame *> frames = framebatch.frames();
    for (int i = 0; i < frames.size(); ++i) {
        Frame *frame = frames[i];
        ::dg::model::WitnessResult *result = batchResponse->add_results();
        string uri = imgDesc[i].data().uri();
        result->mutable_image()->mutable_data()->set_uri(uri);
        result->mutable_image()->mutable_data()->set_height(frame->payload()->data().rows);
        result->mutable_image()->mutable_data()->set_width(frame->payload()->data().cols);
        err = getRecognizeResult(frame, result);
        if (err.code() != 0) {
            LOG(ERROR) << "get result from frame failed, " << err.message();
            return err;
        }
    }

    if (frames.size() != batchResponse->results().size()) {
        LOG(ERROR) << "Input frame size not equal to results size." << frames.size() << "-"
            << batchResponse->results().size() << endl;
        err.set_code(-1);
        err.set_message("Input frame size not equal to results size.");
        return err;
    }

    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Parse batch results cost: " << TimeCostInMs(start, end) << endl;

    gettimeofday(&curr_time, NULL);
    ctx->mutable_responsets()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_responsets()->set_nanosecs((int64_t) curr_time.tv_usec);


    bool storageEnabled = (bool) config_->Value(STORAGE_ENABLED);
    if (storageEnabled) {
        string storageAddress;
        if (batchRequest->context().has_storage()) {
            storageAddress = (string) batchRequest->context().storage().address();
        } else {
            storageAddress = (string) config_->Value(STORAGE_ADDRESS);
        }
        int dbTypeInt = (int) config_->Value(STORAGE_DB_TYPE);
        DBType dbType;
        switch (dbTypeInt) {
            case 0:
                dbType = KAFKA;
                break;
            default:
                dbType = KAFKA;
                break;
        }
        for (int k = 0; k < batchResponse->results_size(); k++) {
            const WitnessResult &r = batchResponse->results(k);
            if (r.vehicles_size() != 0) {
                unique_lock<mutex> lock(WitnessBucket::Instance().mt_push);
                shared_ptr<WitnessVehicleObj> client_request_obj(new WitnessVehicleObj);
                client_request_obj->mutable_storage()->set_address(storageAddress);
                for (int i = 0; i < r.vehicles_size(); i++) {
                    Cutboard c = r.vehicles(i).img().cutboard();
                    Mat roi(framebatch.frames()[k]->payload()->data(), Rect(c.x(), c.y(), c.width(), c.height()));
                    RecVehicle *v = client_request_obj->mutable_vehicleresult()->add_vehicle();
                    v->CopyFrom(r.vehicles(i));
                    bool enablecutboard = (bool) config_->Value("EnableCutboard");
                    if (enablecutboard) {
                        vector<uchar> data(roi.datastart, roi.dataend);
                        string imgdata = Base64::Encode(data);
                        v->mutable_img()->mutable_img()->set_bindata(imgdata);
                    }
                    client_request_obj->mutable_vehicleresult()->mutable_metadata()->CopyFrom(srcMetadatas[k]);
                    client_request_obj->mutable_vehicleresult()->mutable_img()->set_uri(batchRequest->images(k).data().uri());

                    client_request_obj->mutable_vehicleresult()->mutable_img()->set_height(framebatch.frames()[k]->payload()->data().rows);
                    client_request_obj->mutable_vehicleresult()->mutable_img()->set_width(framebatch.frames()[k]->payload()->data().cols);
                }
                WitnessBucket::Instance().Push(client_request_obj);
                lock.unlock();
            }
        }
    }


    VLOG(VLOG_SERVICE) << "Finish batch processing: " << sessionid << "..." << endl;
    return err;
}
MatrixError WitnessAppsService::Index(const IndexRequest *request,
                                      IndexResponse *response) {
    MatrixError err;
    bool gpuplate = (bool) config_->Value(IS_GPU_PLATE);
    switch (request->indextype()) {
        case INDEX_CAR_TYPE:
            for (int i = 0; i < vehicle_repo_.size(); i++) {
                string value =
                    vehicle_repo_[i].brand() + " " + vehicle_repo_[i].subbrand() + " " + vehicle_repo_[i].modelyear();
                (*response->mutable_index())[i] = value;
            }
            break;
        case INDEX_CAR_MAIN_BRAND:
            for (int i = 0; i < vehicle_repo_.size(); i++) {
                string value = vehicle_repo_[i].brand();
                (*response->mutable_index())[i] = value;
            }
            break;
        case INDEX_CAR_SUB_BRAND:
            for (int i = 0; i < vehicle_repo_.size(); i++) {
                string value = vehicle_repo_[i].subbrand();
                (*response->mutable_index())[i] = value;
            }
            break;
        case INDEX_CAR_YEAR_MODEL:
            for (int i = 0; i < vehicle_repo_.size(); i++) {
                string value = vehicle_repo_[i].modelyear();
                (*response->mutable_index())[i] = value;
            }
            break;
        case INDEX_CAR_PLATE_COLOR:
            if (!gpuplate) {
                for (int i = 0; i < plate_color_repo_.size(); i++) {
                    string value = plate_color_repo_[i].data();
                    (*response->mutable_index())[i] = value;
                }
            } else {
                for (int i = 0; i < plate_color_gpu_repo_.size(); i++) {
                    string value = plate_color_gpu_repo_[i].data();
                    (*response->mutable_index())[i] = value;
                }
            }
            break;
        case INDEX_CAR_PLATE_TYPE:
            for (int i = 0; i < plate_type_repo_.size(); i++) {
                string value = plate_type_repo_[i].data();
                (*response->mutable_index())[i] = value;
            }
            break;
        case INDEX_CAR_COLOR:
            for (int i = 0; i < color_repo_.size(); i++) {
                string value = color_repo_[i].data();
                (*response->mutable_index())[i] = value;
            }
            break;
        case INDEX_CAR_MARKER:
            for (int i = 0; i < symbol_repo_.size(); i++) {
                string value = symbol_repo_[i].data();
                (*response->mutable_index())[i] = value;
            }
            break;
        case INDEX_CAR_PEDESTRIAN_ATTR_TYPE:
            for (int i = 0; i < pedestrian_attr_type_repo_.size(); i++) {
                string value = pedestrian_attr_type_repo_[i].data();
                (*response->mutable_index())[i] = value;
            }
            break;
    }

    return err;
}
}

