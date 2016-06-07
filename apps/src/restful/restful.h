/*============================================================================
 * File Name   : restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RESTFUL_H_
#define MATRIX_APPS_RESTFUL_H_

#include <string>

#include "debug_util.h"
#include "pbjson/pbjson.hpp" //from pbjson
#include "Simple-Web-Server/server_http.hpp" //from Simple-Web-Server
#include "../model/common.pb.h"

#include "services/engine_service.h"
#include "log/log_val.h"

using namespace std;
using namespace ::dg::model;
namespace dg {

template<class request_type, class response_type>
using BindFunction = std::function<MatrixError(const request_type *, response_type *)>;

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;

template<class EngineType>
class RestfulService {
public:
    RestfulService(MatrixEnginesPool<EngineType> *engine_pool, Config config,
                   string protocol = "HTTP/1.1",
                   string mime_type =
                   "application/json; charset=utf-8")
        : engine_pool_(engine_pool),
          config_(config),
          protocol_(protocol),
          mime_type_(mime_type) {
    }

    virtual ~RestfulService() {
    }

    void Run() {
        int port = (int) config_.Value("System/Port");
        int gpuNum = (int) config_.Value("System/GpuNum");
        gpuNum = gpuNum == 0 ? 1 : gpuNum;

        int threadsPerGpu = (int) config_.Value("System/ThreadsPerGpu");
        threadsPerGpu = threadsPerGpu == 0 ? 1 : threadsPerGpu;

        int threadNum = gpuNum * threadsPerGpu;

        SimpleWeb::Server<SimpleWeb::HTTP> server(port, threadNum);  //at port with 1 thread
        Bind(server);
        if (engine_pool_ == NULL) {
            LOG(ERROR) << "Engine pool not initialized" << endl;
        }
        engine_pool_->Run();
        cout << typeid(EngineType).name() << " Server(RESTFUL) listening on " << port << endl;
        server.start();
    }


    virtual void Bind(HttpServer &server) = 0;

protected:
    Config config_;
    string protocol_;
    string mime_type_;
    MatrixEnginesPool<EngineType> *engine_pool_;

    static void responseText(HttpServer::Response &response, int code,
                             const string &text) {
        response << "HTTP/1.1 " << std::to_string(code)
            << "\r\nContent-Length: " << text.length()
            << "\r\nContent-Type: application/json; charset=utf-8\r\n\r\n"
            << text;
    }

    template<class apps_type, class request_type, class response_type>
    void bindFunc(HttpServer &server,
                  string endpoint,
                  string method,
                  MatrixError(*func)(apps_type *, const request_type *, response_type *)) {

        server.resource[endpoint][method] =
            [this, func](HttpServer::Response &response, std::shared_ptr<HttpServer::Request> request) {

              VLOG(VLOG_SERVICE) << "[RESTFUL] ========================" << endl;
              VLOG(VLOG_SERVICE) << "[RESTFUL] Get request, thread id: " << this_thread::get_id() << endl;
              struct timeval start, end;
              gettimeofday(&start, NULL);

              request_type protobufRequestMessage;
              response_type protobufResponseMessage;

              try {
                  string content = request->content.string();
                  string err;
                  int ret = pbjson::json2pb(content, &protobufRequestMessage, err);
                  if (ret < 0) {
                      responseText(response, 400, "parameter conversion failed: " + err);
                      return;
                  }
                  CallData data;
                  data.func = [func, &protobufRequestMessage, &protobufResponseMessage, &data]() -> MatrixError {
                    return (bind(func, (apps_type *) data.apps,
                                 placeholders::_1,
                                 placeholders::_2))(&protobufRequestMessage,
                                                    &protobufResponseMessage);
                  };

                  if (engine_pool_ == NULL) {
                      LOG(ERROR) << "Engine pool not initailized. " << endl;
                      return;
                  }

                  engine_pool_->enqueue(&data);

                  MatrixError error = data.Wait();

                  if (error.code() != 0) {
                      responseText(response, 500, error.message());
                      return;
                  }

                  content = "";
                  pbjson::pb2json(&protobufResponseMessage, content);
                  responseText(response, 200, content);

                  gettimeofday(&end, NULL);
                  VLOG(VLOG_PROCESS_COST) << "[RESTFUL] Total cost: " << TimeCostInMs(start, end) << endl;
                  VLOG(VLOG_SERVICE) << "[RESTFUL] ========================" << endl;
              }
              catch (exception &e) {
                  responseText(response, 500, e.what());
              }
            };
    }

};

}

#endif //MATRIX_APPS_RESTFUL_H_
