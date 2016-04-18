/*============================================================================
 * File Name   : ranker_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "timing_profiler.h"
#include "codec/base64.h"
#include "io/reader.h"

#include "util/feature_serializer.h"
#include "alg/car_matcher.h"
#include "alg/car_ranker.h"
#include "model/simservice.grpc.pb.h"

using namespace cv;
using namespace std;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using model::FeatureRankingRequest;
using model::FeatureRankingResponse;
using model::SimilarityService;
using model::Image;

class RankerServiceImpl final : public SimilarityService::Service
{
    
    virtual Status GetRankedFaceVector(ServerContext* context, const FeatureRankingRequest* request, FeatureRankingResponse* response) override
    {

        return Status();
    }

    virtual Status GetRankedCarVector(ServerContext* context, const FeatureRankingRequest* request,
            FeatureRankingResponse* response) override
    {
        try
        {
            google::protobuf::int64 reqid = request->reqid();
            stringstream reqid_ss;
            reqid_ss << reqid;
            string reqid_str = reqid_ss.str();
            LOG(INFO)<< "Received a request, reqid=" + reqid_str;

            response->set_reqid(reqid);

            vector<CarDescriptor> all_des;
            FeatureSerializer serializer;

            LOG(INFO)<< "Candidates size: " << request->candidates_size();

            for (int i = 0; i < request->candidates_size(); i++)
            {
                try
                {
                    Mat desmat, posmat;
                    string feature = request->candidates(i).feature();
                    // LOG(INFO)<< feature << " size: " <<feature.size();
                    if (feature.size() <= 0)
                    {
                        LOG(WARNING)<< "Received a bad request, reqid=" + reqid_str + ", candidate is invalid";
                        return Status::CANCELLED;
                    }
                    serializer.FeatureDeserialize(feature, desmat, posmat);
                    CarDescriptor des;
                    des.descriptor = desmat;
                    des.position = posmat;
                    all_des.push_back(des);
                }
                catch (const std::exception& e)
                {
                    LOG(WARNING)<< "Received a bad request, reqid=" + reqid_str + ", index="<<i<<", " + string(e.what());
                    return Status::CANCELLED;
                }
            }

            LOG(INFO)<< "Features unserialize finished";

            vector<uchar> jpgdata;
            string imgdata = request->image().bindata();
            if (imgdata.size() <= 0)
            {
                LOG(WARNING)<< "Received a bad request, reqid=" + reqid_str + ", bindata is invalid";
                return Status::CANCELLED;
            }

            Mat img;
            try
            {
                Base64::Decode(imgdata, jpgdata);
                img = imdecode(Mat(jpgdata), 1);
            }
            catch (const std::exception& e)
            {
                LOG(WARNING)<< "Received a bad request, reqid=" + reqid_str + ", " + string(e.what());
                return Status::CANCELLED;
            }

            LOG(INFO)<< "Target picture unserialize finished";

            Rect selected_box;
            if (request->interestedareas_size() > 0)
            {
                selected_box.x = 0;
                selected_box.y = 0;
                selected_box.width = img.cols;
                selected_box.height = img.rows;
            }
            else
            {
                if (request->interestedareas(0).width() == 0
                        || request->interestedareas(0).height() == 0)
                {
                    LOG(WARNING)<< "Received a bad request, reqid=" + reqid_str + ", interestedarea is invalid, rankerd will use full size";
                    selected_box.x = 0;
                    selected_box.y = 0;
                    selected_box.width = img.cols;
                    selected_box.height = img.rows;
                }
                else
                {
                    selected_box.x = request->interestedareas(0).x();
                    selected_box.y = request->interestedareas(0).y();
                    selected_box.width = request->interestedareas(0).width();
                    selected_box.height = request->interestedareas(0).height();
                }
            }
            LOG(INFO)<<"Pic size: "<<img.cols<<"*"<<img.rows;
            LOG(INFO)<< "Interested areas unserialize finished";
            int loopsize =
                    (request->limit() <= 0 || request->limit() >= all_des.size()) ?
                            all_des.size() : request->limit();

            vector<CarScore> topx;
            try
            {
                topx = ranker.Rank(selected_box, img, all_des, loopsize);
            }
            catch (const std::exception& e)
            {
                LOG(WARNING)<< "Received a bad request, reqid=" + reqid_str + ", " + string(e.what());
                return Status::CANCELLED;
            }

            for (int i = 0; i < topx.size(); i++)
            {
                CarScore cs = topx[i];
                response->add_ids(request->candidates(cs.index).id());
                response->add_scores(cs.score);
            }

            LOG(INFO)<< "Rank finished";
            return Status::OK;
        }
        catch (const std::exception& e)
        {
            LOG(WARNING)<< "Received a bad request ," + string(e.what());
            return Status::CANCELLED;
        }
    }

    CarRanker ranker;
};

void RunServer(string port)
{
    std::string server_address("0.0.0.0:" + port);
    RankerServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    // builder.SetMaxMessageSize(524288000);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    LOG(INFO)<< "Server listening on " << server_address;
    server->Wait();
}

// int main(int argc, char *argv[])
// {
//     if (argc < 2)
//     {
//         printf("Usage: %s [port] [glog args]\n", argv[0]);
//     }

//     google::InitGoogleLogging("rankerd");
//     google::ParseCommandLineFlags(&argc, &argv, true);
//     RunServer(string(argv[1]));
//     return 0;
// }

