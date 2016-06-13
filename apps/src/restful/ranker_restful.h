/*============================================================================
 * File Name   : ranker_restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RESTFUL_RANKER_H_
#define MATRIX_APPS_RESTFUL_RANKER_H_
#include "restful.h"
#include "services/ranker_service.h"
#include "services/system_service.h"

namespace dg {


typedef MatrixError (*RankFunc)(RankerAppsService *, const FeatureRankingRequest *, FeatureRankingResponse *);
typedef MatrixError (*PingFunc)(SystemAppsService *, const PingRequest *, PingResponse *);
//typedef MatrixError (*BatchRecFunc)(WitnessAppsService *, const WitnessBatchRequest *, WitnessBatchResponse *);


class RestRankerServiceImpl final: public RestfulService<RankerAppsService> {

public:

    RestRankerServiceImpl(Config config,
                          string addr,
                          MatrixEnginesPool <RankerAppsService> *engine_pool)
        : RestfulService(engine_pool, config), service_system_(&config, "system"), config_(config) {
    }

    virtual ~RestRankerServiceImpl() { }

    void Bind(HttpServer &server) {

        RankFunc rank_func = (RankFunc) &RankerAppsService::GetRankedVector;
        bindFunc<RankerAppsService, FeatureRankingRequest, FeatureRankingResponse>(server,
                                                                                   "^/rank$",
                                                                                   "POST",
                                                                                   rank_func);
        std::function<MatrixError(const SystemStatusRequest *, SystemStatusResponse *)> statusBinder =
            std::bind(&SystemAppsService::SystemStatus, &service_system_, std::placeholders::_1, std::placeholders::_2);
        bind1(server, "^/info$", "GET", statusBinder);
        std::function<MatrixError(const PingRequest *, PingResponse *)> pingBinder =
            std::bind(&SystemAppsService::Ping, &service_system_, std::placeholders::_1, std::placeholders::_2);
        bind1(server, "^/ping$", "GET", pingBinder);

    }


private:
    SystemAppsService service_system_;

    Config config_;
};

}

#endif //MATRIX_APPS_RESTFUL_RANKER_H_
