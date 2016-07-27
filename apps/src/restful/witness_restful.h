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
#include "services/repo_service.h"

namespace dg {

typedef MatrixError (*RecFunc)(WitnessAppsService *, const WitnessRequest *, WitnessResponse *);
typedef MatrixError (*BatchRecFunc)(WitnessAppsService *, const WitnessBatchRequest *, WitnessBatchResponse *);
typedef MatrixError (*RecIndexFunc)(WitnessAppsService *, const IndexRequest *, IndexResponse *);
typedef MatrixError (*RecIndexTxtFunc)(WitnessAppsService *, const IndexTxtRequest *, IndexTxtResponse *);
class RestWitnessServiceImpl final: public RestfulService {
public:
    RestWitnessServiceImpl(Config config,
                           string addr)
        : RestfulService(config) {
        service_ = new WitnessAppsService(&config, "WitnessAppsService");
    }

    virtual ~RestWitnessServiceImpl() { delete service_; }

    void Bind(HttpServer &server) {

        std::function<MatrixError(const WitnessRequest *, WitnessResponse *)> recBinder =
            std::bind(&WitnessAppsService::Recognize, service_, std::placeholders::_1, std::placeholders::_2);
        bindFunc<WitnessRequest, WitnessResponse>(server, "^/rec/image$",
                                                  "POST", recBinder);
        std::function<MatrixError(const WitnessBatchRequest *, WitnessBatchResponse *)> recBatchBinder =
            std::bind(&WitnessAppsService::BatchRecognize, service_, std::placeholders::_1, std::placeholders::_2);
        bindFunc<WitnessBatchRequest, WitnessBatchResponse>(server,
                                                            "/rec/image/batch$",
                                                            "POST",
                                                            recBatchBinder);


        std::function<MatrixError(const IndexRequest *, IndexResponse *)> indexBinder =
            std::bind(&RepoService::Index, RepoService::GetInstance(), std::placeholders::_1, std::placeholders::_2);
        bindFunc<IndexRequest, IndexResponse>(server, "^/rec/index$", "POST", indexBinder);

        std::function<MatrixError(const IndexTxtRequest *, IndexTxtResponse *)> indexTxtBinder =
            std::bind(&RepoService::IndexTxt, RepoService::GetInstance(), std::placeholders::_1, std::placeholders::_2);
        bindFunc<IndexTxtRequest, IndexTxtResponse>(server, "^/rec/index/txt$", "POST", indexTxtBinder);

    }

protected:
    WitnessAppsService *service_;
};
}

#endif //MATRIX_APPS_RESTFUL_WITNESS_H_
