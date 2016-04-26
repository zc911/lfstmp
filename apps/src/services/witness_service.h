/*============================================================================
 * File Name   : ranker_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_WITNESS_SERVICE_H_
#define MATRIX_APPS_WITNESS_SERVICE_H_

#include "model/witness.grpc.pb.h"

namespace dg 
{

class WitnessService
{
public:
    WitnessService(Config *config);
    virtual ~WitnessService();

    bool Recognize(const WitnessRequest *request, WitnessResponse *response);

    bool BatchRecognize(const WitnessBatchRequest *request, WitnessBatchResponse *response);

private:
    Config *config_;

};

}

#endif //MATRIX_APPS_WITNESS_SERVICE_H_