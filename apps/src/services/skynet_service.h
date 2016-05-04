/*============================================================================
 * File Name   : skynet_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_SKYNET_SERVICE_H_
#define MATRIX_APPS_SKYNET_SERVICE_H_

#include "config.h"
#include "model/skynet.grpc.pb.h"
namespace dg
{
using namespace ::dg::model;

class SkynetAppsService
{

public:
    SkynetAppsService(const Config *config);
    virtual ~SkynetAppsService();

    bool VideoRecognize(const SkynetRequest *request, SkynetResponse *response);
    
private:
    const Config *config_;

};

}

#endif //MATRIX_APPS_SKYNET_SERVICE_H_
