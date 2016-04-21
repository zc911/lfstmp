/*============================================================================
 * File Name   : grpc_main.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/


#include <glog/logging.h>
#include <grpc++/grpc++.h>

#include "codec/base64.h"
#include "model/simservice.grpc.pb.h"
#include "service/ranker_service.h"

using namespace model;
using namespace dg;
using namespace cv;
using namespace std;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

int _USER_ERROR__missing_dlib_all_source_cpp_file__OR__inconsistent_use_of_DEBUG_or_ENABLE_ASSERTS_preprocessor_directives_;

class RankerServiceImpl final : public model::SimilarityService::Service
{
private:
    CarRankService car_ranker_;
    FaceRankService face_ranker_;

    template <typename F>
    bool process(const FeatureRankingRequest* request, RankService<F>& ranker,  FeatureRankingResponse* response)
    {
        google::protobuf::int64 reqid = request->reqid();
        LOG(INFO) << "request(" << reqid << ")" << endl;
        response->set_reqid(reqid);

        if (!request->has_image()) {
            LOG(WARNING) << "no image in request context" << endl;
            return false;
        }

        if (request->candidates_size() <= 0) {
            LOG(WARNING) << "no candidates in request context" << endl;
            return false;
        }

        string imgdata = request->image().bindata();
        if (imgdata.size() <= 0)
        {
            LOG(WARNING) << "bad request(" << reqid << "), invalid image." << endl;
            return false;
        }
        vector<uchar> jpgdata;
        Base64::Decode(imgdata, jpgdata);

        Mat image = imdecode(Mat(jpgdata), 1);

        int limit = request->limit();
        if (limit <= 0 || limit >= request->candidates_size())
        {
            limit = request->candidates_size();
        }

        vector<F> features;
        for(int i = 0; i < request->candidates_size(); i++)
        {
            string featureStr = request->candidates(i).feature();
            if (featureStr.size() <= 0)
            {
                LOG(WARNING) << "bad request(" << reqid << "), invalid candidate" << endl;
                return false;
            }

            F feature = ranker.Deserialize(featureStr);
            features.push_back(feature);
        }

        Rect hotspot(0, 0, image.cols, image.rows);
        if ( request->interestedareas_size() > 0 )
        {
            const Cutboard& cb = request->interestedareas(0);
            if (cb.width() != 0 && cb.height() != 0)
            {
                hotspot = Rect(cb.x(), cb.y(), cb.width(), cb.height());
            }
        }

        vector<Score> topn = ranker.Rank(image, hotspot, features);
        //sort
        partial_sort(topn.begin(), topn.begin() + limit, topn.end());
        topn.resize(limit);

        for (int i = 0; i < topn.size(); i ++)
        {
            Score& s = topn[i];
            response->add_ids(request->candidates(s.index).id());
            response->add_scores(s.score);
        }

        return true;
    }

    virtual Status GetRankedVector(ServerContext* context, const FeatureRankingRequest* request, FeatureRankingResponse* response) override
    {
        try
        {
            return process(request, car_ranker_, response) ? Status::OK : Status::CANCELLED;
        }
        catch (const std::exception& e)
        {
            LOG(WARNING) << "bad request(" << request->reqid() << "), " << e.what() << endl;
            return Status::CANCELLED;
        }
    }


    // virtual Status GetRankedCarVector(ServerContext* context, const FeatureRankingRequest* request, FeatureRankingResponse* response) override
    // {
    //     try
    //     {
    //         return process(request, car_ranker_, response) ? Status::OK : Status::CANCELLED;
    //     }
    //     catch (const std::exception& e)
    //     {
    //         LOG(WARNING) << "bad request(" << request->reqid() << "), " << e.what() << endl;
    //         return Status::CANCELLED;
    //     }
    // }


    
    // virtual Status GetRankedFaceVector(ServerContext* context, const FeatureRankingRequest* request, FeatureRankingResponse* response) override
    // {
    //     try
    //     {
    //         return process(request, face_ranker_, response) ? Status::OK : Status::CANCELLED;
    //     }
    //     catch (const std::exception& e)
    //     {
    //         LOG(WARNING) << "bad request(" << request->reqid() << "), " << e.what() << endl;
    //         return Status::CANCELLED;
    //     }

    //     return Status::CANCELLED;
    // }
};

void RunServer(string address)
{
    RankerServiceImpl service;
    ServerBuilder builder;

    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    LOG(INFO)<< "Server listening on " << address;
    server->Wait();
}

int main(int argc, char* argv[])
{
    string server("0.0.0.0:9876");
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " [address] [glog args]" << endl;
    }
    else
    {
        server = string(argv[1]);
        if (server.size() == 0)server = string("0.0.0.0:9876");
    }

    google::InitGoogleLogging("rankerd");
    google::ParseCommandLineFlags(&argc, &argv, true);
    RunServer(server);
    return 0;
}