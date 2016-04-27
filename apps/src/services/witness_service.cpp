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
#include "codec/base64.h"
#include "ranker_service.h"
#include "image_service.h"
 

namespace dg
{

WitnessAppsService::WitnessAppsService(Config *config)
                    : config_(config)
{

}

WitnessAppsService::~WitnessAppsService()
{

}

bool WitnessAppsService::Recognize(const WitnessRequest *request, WitnessResponse *response)
{

    return true;
}

bool WitnessAppsService::BatchRecognize(const WitnessBatchRequest *request, WitnessBatchResponse *response)
{

    return true;
}

}
