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

namespace dg
{

class SkynetService
{

public:
    SkynetService(Config *config);
    virtual ~SkynetService();

    bool VideoRecognize(const SkynetRequest *request, SkynetResponse *response);
    
private:
    Config *config_;

};

}

#endif //MATRIX_APPS_SKYNET_SERVICE_H_