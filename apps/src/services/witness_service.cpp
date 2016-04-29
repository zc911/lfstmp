/*============================================================================
 * File Name   : witness_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <glog/logging.h>

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

        Identification curr_id = id_ ++;
        Frame frame(curr_id, image);

        Operation op;
        op.Set(OPERATION_VEHICLE_DETECT);
        frame.set_operation(op);

        FrameBatch framebatch(curr_id * 10, 1);
        framebatch.add_frame(&frame);
        engine_.Process(&framebatch);

        for(const Object *o : frame.objects())
        {

        }

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
