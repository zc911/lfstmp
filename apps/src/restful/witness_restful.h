/*============================================================================
 * File Name   : witness_restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RESTFUL_WITNESS_H_
#define MATRIX_APPS_RESTFUL_WITNESS_H_

#include "restful.h"
#include "services/witness_service.h"
#include "services/system_service.h"

namespace dg {

typedef MatrixError (*RecFunc)(WitnessAppsService *, const WitnessRequest *, WitnessResponse *);
typedef MatrixError (*BatchRecFunc)(WitnessAppsService *, const WitnessBatchRequest *, WitnessBatchResponse *);
typedef MatrixError (*RecIndexFunc)(WitnessAppsService *, const IndexRequest *, IndexResponse *);
typedef MatrixError (*RecIndexTxtFunc)(WitnessAppsService *, const IndexTxtRequest *, IndexTxtResponse *);
class RestWitnessServiceImpl final: public RestfulService<WitnessAppsService,WitnessEngine> {
public:
    RestWitnessServiceImpl(Config config,
                           string addr,
                           ServicePool<WitnessAppsService,WitnessEngine> *service_pool)
        : RestfulService(service_pool, config){

    }

    virtual ~RestWitnessServiceImpl() { }

    void Bind(HttpServer &server) {

        RecFunc rec_func = (RecFunc) &WitnessAppsService::Recognize;
        bindFunc<WitnessAppsService, WitnessRequest, WitnessResponse>(server, "^/rec/image$",
                                                                      "POST", rec_func);
        BatchRecFunc batch_func = (BatchRecFunc) &WitnessAppsService::BatchRecognize;
        bindFunc<WitnessAppsService, WitnessBatchRequest, WitnessBatchResponse>(server,
                                                                                "/rec/image/batch$",
                                                                                "POST",
                                                                                batch_func);


        RecIndexFunc rec_index_func = (RecIndexFunc) &WitnessAppsService::Index;
        bindFunc<WitnessAppsService, IndexRequest, IndexResponse>(server, "^/rec/index$",
                                                                  "POST", rec_index_func);
        RecIndexTxtFunc rec_index_txt_func = (RecIndexTxtFunc) &WitnessAppsService::IndexTxt;
        bindFunc<WitnessAppsService, IndexTxtRequest, IndexTxtResponse>(server, "^/rec/index/txt$",
                                                                        "POST", rec_index_txt_func);

    }

protected:

};
}

#endif //MATRIX_APPS_RESTFUL_WITNESS_H_
