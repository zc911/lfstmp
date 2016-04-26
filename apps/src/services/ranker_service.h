/*============================================================================
 * File Name   : ranker_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RANKER_SERVICE_H_
#define MATRIX_APPS_RANKER_SERVICE_H_


#include "model/ranker.grpc.pb.h" //from apps


using namespace cv;
using namespace std;

namespace dg 
{

class RankerService
{
public:
    RankerService(Config *config);
    virtual ~RankerService();

    bool GetRankedVector(const FeatureRankingRequest* request, FeatureRankingResponse* response);

private:
    Config *config_;
    CarRankService car_ranker_;
    FaceRankService face_ranker_;


    bool getRankedCarVector(const FeatureRankingRequest* request, FeatureRankingResponse* response);


    bool getRankedFaceVector(const FeatureRankingRequest* request, FeatureRankingResponse* response);

    static void sortAndFillResponse(const FeatureRankingRequest* request, vector<Score>& scores, FeatureRankingResponse* response);

    static string requestPrefix(const FeatureRankingRequest* request);

    static Rect getHotspot(const FeatureRankingRequest* request, const Mat& image);

    static int getLimit(const FeatureRankingRequest* request);

    template <typename F>
    static MatrixError extractFeatures(const FeatureRankingRequest* request, RankService<F>& ranker, vector<F>& features)
    {
        MatrixError err;
        if (request->candidates_size() <= 0)
        {
            err.set_code(-1);
            err.set_message("no candidates in request context");
            return err;
        }

        for(int i = 0; i < request->candidates_size(); i ++)
        {
            string featureStr = request->candidates(i).feature();
            if (featureStr.size() <= 0)
            {
                err.set_code(-1);
                err.set_message("invalid candidate");
                return err;
            }

            features.push_back(ranker.Deserialize(featureStr));
        }

        return err;
    }
};

}

 #endif //MATRIX_APPS_RANKER_SERVICE_H_