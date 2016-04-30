/*============================================================================
 * File Name   : witness_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <glog/logging.h>
#include <sys/time.h>

#include "witness_service.h"
#include "image_service.h"
//#include "ranker_service.h"
 

namespace dg
{

WitnessAppsService::WitnessAppsService(const Config *config)
                    : config_(config)
                    , engine_(*config)
                    , id_(0)
{

}

WitnessAppsService::~WitnessAppsService()
{

}

Operation WitnessAppsService::getOperation(const WitnessRequestContext& ctx)
{
    Operation op;
    for(int i = 0; i < ctx.functions_size(); i ++)
    {
        switch(ctx.functions(i))
        {
        case RECFUNC_NONE: op.Set(OPERATION_NONE); break;
        case RECFUNC_VEHICLE: op.Set(OPERATION_VEHICLE); break;
        case RECFUNC_VEHICLE_DETECT: op.Set(OPERATION_VEHICLE_DETECT); break;
        case RECFUNC_VEHICLE_TRACK: op.Set(OPERATION_VEHICLE_TRACK); break;
        case RECFUNC_VEHICLE_STYLE: op.Set(OPERATION_VEHICLE_STYLE); break;
        case RECFUNC_VEHICLE_COLOR: op.Set(OPERATION_VEHICLE_COLOR); break;
        case RECFUNC_VEHICLE_MARKER: op.Set(OPERATION_VEHICLE_MARKER); break;
        case RECFUNC_VEHICLE_PLATE: op.Set(OPERATION_VEHICLE_PLATE); break;
        case RECFUNC_VEHICLE_FEATURE_VECTOR: op.Set(OPERATION_VEHICLE_FEATURE_VECTOR); break;
        case RECFUNC_FACE: op.Set(OPERATION_FACE); break;
        case RECFUNC_FACE_DETECTOR: op.Set(OPERATION_FACE_DETECTOR); break;
        case RECFUNC_FACE_FEATURE_VECTOR: op.Set(OPERATION_FACE_FEATURE_VECTOR); break;
        default: break;
        }
    }
    return op;
}

void WitnessAppsService::copyCutboard(const Box &b, Cutboard *cb)
{
    cb->set_x(b.x);
    cb->set_y(b.y);
    cb->set_width(b.width);
    cb->set_height(b.height);
}

MatrixError WitnessAppsService::fillModel(Identification id, VehicleModel *model)
{
    MatrixError err;

    model->set_brandid(id);//just set class id to brandid for test

    return err;
}

MatrixError WitnessAppsService::fillColor(const dg::Color &color, model::Color *rcolor)
{
    MatrixError err;

    rcolor->set_id(color.class_id);
    rcolor->set_confidence(color.confidence);

    return err;
}

MatrixError WitnessAppsService::fillPlate(const Plate &plate, LicensePlate *rplate)
{
    MatrixError err;
    rplate->set_platenum(plate.plate_num);
    copyCutboard(plate.box, rplate->mutable_cutboard());
    rplate->set_colorid(plate.color_id);
    rplate->set_typeid(plate.plate_type);
    rplate->set_confidence(plate.confidence);

    return err;
}

MatrixError WitnessAppsService::fillSymbols(const vector<Object*>& objects, RecognizedVehicle *vrec))
{
    MatrixError err;

    int isize = 6;  //TODO: 6 to size
    SymbolItem *items = new SymbolItem[isize]; 
    for(const Object *object : objects)
    {
        LOG(INFO) << "recognized object(marker?): " << object->id() << ", type: " << object->type();
        if (object->type() != OBJECT_MARKER)
        {
            LOG(WARNING) << "unknown marker type: " << object->type();
            continue;
        }

        Marker *m = (Marker *)object;
        Identification mid = m->class_id();
        if (mid >= 0 && mid < isize)
        {
            Symbol *s =  items[mid]->mutable_symbols();
            s->set_confidence(m->detection().confidence);
            copyCutboard(m->detection().box, s->mutable_cutboard());
        }
    }

    for(int i = 0; i < isize; i ++)
    {
        SymbolItem *item = items[i];
        if (item->symbols_size() > 0)
        {
            SymbolItem *addItem = vrec->add_symbolitems();
            addItem->CopyFrom(*item);
        }
    }

    return err;
}

MatrixError WitnessAppsService::getRecognizedVehicle(Vehicle *vobj, RecognizedVehicle *vrec)
{
    MatrixError err;
    vrec->set_features(vobj->feature().Serialize());

    const Detection &d = vobj->window();
    LOG(INFO) << "detection id: " << d.id << ", deleted? " << d.deleted;
    copyCutboard(d.box, vrec->mutable_cutboard());

    err = fillModel(vobj->class_id(), vrec->mutable_model());
    if (err.code < 0)return err;

    err = fillColor(vobj->color(), vrec->mutable_color());
    if (err.code < 0)return err;

    err = fillPlate(vobj->plate(), vrec->mutable_licenseplate());
    if (err.code < 0)return err;

    err = fillSymbols(vobj->children(), vrec);
    if (err.code < 0)return err;

    return err;
}

MatrixError WitnessAppsService::getRecognizedFace(Face *fobj, RecognizedFace *frec)
{
    MatrixError err;
    frec->set_confidence((float)fobj->confidence());
    frec->set_features(fobj->feature().Serialize());

    const Detection &d = fobj->detection();
    LOG(INFO) << "detection id: " << d.id << ", deleted? " << d.deleted;
    copyCutboard(d.box, frec->mutable_cutboard());
    return err;
}

MatrixError WitnessAppsService::getRecognizeResult(const Frame *frame, WitnessResult *result)
{
    MatrixError err;

    for(const Object *object : frame->objects())
    {
        LOG(INFO) << "recognized object: " << object->id() << ", type: " << object->type();
        switch(object->type())
        {
        case OBJECT_CAR:
            RecognizedVehicle *vehicle = result->add_vehicles();
            err = getRecognizedVehicle((Vehicle *)object, vehicle)

        case OBJECT_FACE:
            RecognizedFace *face = result->add_faces();
            err = getRecognizedFace((Face *)object, face);
            
        default:
            LOG(WARNING) << "unknown object type: " << object->type();
        }

        if (err.code < 0)
        {
            break;
        }
    }

    return err;
}

bool WitnessAppsService::Recognize(const WitnessRequest *request, WitnessResponse *response)
{
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    
    const string& sessionid = request->context().sessionid();

    if (!request->has_image() || !request->image().has_data())
    {
        LOG(ERROR) << "image descriptor does not exist";
        return false;
    }

    LOG(INFO) << "Get Recognize request: " << sessionid
              << ", Image URI:" << request->image().data().uri();
    LOG(INFO) << "Start processing: " << sessionid << "...";

    Mat image;
    MatrixError err = ImageService::ParseImage(request->image().data(), image);
    if (err.code() != 0)
    {
        LOG(ERROR) << "parse image failed, " << err.message();
        return false;
    }

    Identification curr_id = id_ ++; //TODO: make thread safe
    Frame *frame = new Frame(curr_id, image);
    frame->set_operation(getOperation(request->context()));

    FrameBatch framebatch(curr_id * 10, 1);
    framebatch.add_frame(frame);
    engine_.Process(&framebatch);

    //fill response
    ::dg::WitnessResponseContext* ctx = response->mutable_context();
    ctx->set_sessionid(sessionid);
    ctx->mutable_requestts()->set_seconds((int64_t)curr_time.tv_sec);
    ctx->mutable_requestts()->set_nanosecs((int64_t)curr_time.tv_usec);
    ctx->set_status("200");
    ctx->set_message("SUCCESS");

    //debug information of this request
    ::google::protobuf::Map<::std::string, ::dg::Time>& debugTs = *ctx->mutable_debugts();
    debugTs["start"] = ctx->requestts();


    WitnessResult *result = response->mutable_result();
    err = getRecognizeResult(frame, result);
    if (err.code() != 0)
    {
        LOG(ERROR) << "get result from frame failed, " << err.message();
        return false;
    }

    WitnessImage *ret_image = result->mutable_image();
    ret_image->CopyFrom(request->image());
    ret_image->data().set_width(image.cols);
    ret_image->data().set_height(image.rows);
    //if ReturnsImage, compress image into data.bindata

    gettimeofday(&curr_time, NULL);
    ctx->mutable_responsets()->set_seconds((int64_t)curr_time.tv_sec);
    ctx->mutable_responsets()->set_nanosecs((int64_t)curr_time.tv_usec);
    debugTs["end"] = ctx->responsets();

    cout << "recognized objects: " << frame->objects().size() << endl;

    cout << "Finish processing: " << sessionid << "..." << endl;
    cout << "=======" << endl;
    return true;
}

bool WitnessAppsService::BatchRecognize(const WitnessBatchRequest *request, WitnessBatchResponse *response)
{

    return true;
}

}
