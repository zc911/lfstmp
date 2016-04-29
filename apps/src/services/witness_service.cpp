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

bool WitnessAppsService::Recognize(const WitnessRequest *request, WitnessResponse *response)
{
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    
    const string& sessionid = request->context().sessionid();
    cout << "Get Recognize request: " << sessionid
         << ", Image URI:" << request->image().data().uri() << endl;
    cout << "Start processing: " << sessionid << "..." << endl;

    if (!request->has_image() || !request->image().has_data())
    {
        LOG(ERROR) << "image descriptor does not exist";
        return false;
    }

    Mat image;
    MatrixError err = ImageService::ParseImage(request->image().data(), image);
    if (err.code() != 0)
    {
        LOG(ERROR) << "parse image failed, " << err.message();
        return false;
    }

    Identification curr_id = id_ ++; //TODO: make thread safe
    Frame frame(curr_id, image);

    Operation op;
    for(int i = 0; i < request->context().functions_size(); i ++)
    {
        switch(request->context().functions(i))
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
    frame.set_operation(op);

    FrameBatch framebatch(curr_id * 10, 1);
    framebatch.add_frame(&frame);
    engine_.Process(&framebatch);

    ::dg::WitnessResponseContext* ctx = response->mutable_context();
    ctx->set_sessionid(request->context().sessionid());
    ctx->mutable_requestts()->set_seconds((int64_t)curr_time.tv_sec);
    ctx->mutable_requestts()->set_nanosecs((int64_t)curr_time.tv_usec);
    ctx->set_status("200");
    ctx->set_message("SUCCESS");

    ::google::protobuf::Map<::std::string, ::dg::Time>* debugTs = ctx->mutable_debugts()
    (*debugTs)["start"] = ctx->requestts();

    for(const Object *o : frame.objects())
    {
        switch(o->type())
        {
        case OBJECT_CAR:

        case OBJECT_FACE:
            
        default:
            LOG(WARNING) << "unknown object type: " << o->type();
        }
    }

    gettimeofday(&curr_time, NULL);
    ctx->mutable_responsets()->set_seconds((int64_t)curr_time.tv_sec);
    ctx->mutable_responsets()->set_nanosecs((int64_t)curr_time.tv_usec);
    (*debugTs)["end"] = ctx->responsets();

    cout << "recognized objects: " << frame.objects().size() << endl;

    cout << "Finish processing: " << sessionid << "..." << endl;
    cout << "=======" << endl;
    return true;
}

bool WitnessAppsService::BatchRecognize(const WitnessBatchRequest *request, WitnessBatchResponse *response)
{

    return true;
}

}
