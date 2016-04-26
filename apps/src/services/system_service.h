/*============================================================================
 * File Name   : system_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_SYSTEM_SERVICE_H_
#define MATRIX_APPS_SYSTEM_SERVICE_H_

namespace dg
{

class SystemService
{

public:
    WitnessService(Config *config);
    virtual ~WitnessService();

    bool Ping(const PingRequest *request, PingResponse *response);

    bool SystemStatus(const SystemStatusRequest *request, SystemStatusResponse *response);

    bool GetInstances(const GetInstancesRequest *request, InstanceConfigureResponse *response);

    bool ConfigEngine(const InstanceConfigureRequest *request, InstanceConfigureResponse *response);
    
private:
    Config *config_;

};
};

}

#endif //MATRIX_APPS_SYSTEM_SERVICE_H_