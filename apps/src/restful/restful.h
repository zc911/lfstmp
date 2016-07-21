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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace ::dg::model;
namespace dg {
enum ErrorCode {
  NoError = 200,
  ServiceError = 500,
  RequestError = 400
};
template<class request_type, class response_type>
using BindFunction = std::function<MatrixError(const request_type *, response_type *)>;

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;
static void responseText(HttpServer::Response &response, int code,
                         const string &text) {
  response << "HTTP/1.1 " << std::to_string(code)
           << "\r\nContent-Length: " << text.length()
           << "\r\nContent-Type: application/json; charset=utf-8\r\n\r\n"
           << text;
}
template<class ServiceType, class EngineType>

class RestfulService {

public:

  RestfulService(ServicePool<ServiceType, EngineType> *service_pool, Config config,
                 string protocol = "HTTP/1.1",
                 string mime_type =
                   "application/json; charset=utf-8")
    : service_pool_(service_pool),
      config_(config),
      protocol_(protocol),
      mime_type_(mime_type),
      sys_apps_(&config, "witness system") {

  }

  virtual ~RestfulService() {
  }

  void Run() {
    int port = (int) config_.Value("System/Port");

    int threadsInTotal = 0;
    int gpuNum = (int) config_.Value(SYSTEM_THREADS + "/Size");
    for (int i = 0; i < gpuNum; ++i) {
      int threadsOnGpu = (int) config_.Value(SYSTEM_THREADS + std::to_string(i));
      threadsInTotal += threadsOnGpu;
    }
    SimpleWeb::Server<SimpleWeb::HTTP> server(port, threadsInTotal * 10);

    Bind(server);
    if (service_pool_ == NULL) {
      LOG(ERROR) << "Engine pool not initialized" << endl;
    }
    service_pool_->Run();
    cout << typeid(ServiceType).name() << " Server(RESTFUL) listening on " << port << endl;
    string instanceType = (string) config_.Value("InstanceType");
    server.start();
  }

  virtual void Bind(HttpServer &server) = 0;

protected:

  Config config_;
  string protocol_;
  string mime_type_;
  ServicePool<ServiceType, EngineType> *service_pool_;


  // This function binds request operation to specific processor
  // There are two bindFunc implmentation and the differents between these two
  // is the later one need to passed into an engine instance since the GPU
  // thread limitation.
  template<class request_type, class response_type>
  void bindFunc(
    HttpServer &server,
    string endpoint,
    string method,
    std::function<MatrixError(const request_type *, response_type *)> func) {

    server.resource[endpoint][method] =
    [func, endpoint](HttpServer::Response & response, std::shared_ptr<HttpServer::Request> request) {
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

        MatrixError error = func(&protobufRequestMessage, &protobufResponseMessage);
        if (error.code() != 0) {
          responseText(response, ServiceError, error.message());
          return;
        }

        content = "";
        pbjson::pb2json(&protobufResponseMessage, content);
        responseText(response, NoError, content);
      }
      catch (exception &e) {
        responseText(response, ServiceError, e.what());
      }
    };

  }

  // This function need to pass into an engine instance since
  // GPU thread limitation. So this function only bind the recognize/batchRecognize/ranker etc.

  template<class apps_type, class request_type, class response_type>
  void bindFunc(HttpServer &server,
                string endpoint,
                string method,
                MatrixError(*func)(apps_type *, const request_type *, response_type *)) {


    server.resource[endpoint][method] =
    [this, func](HttpServer::Response & response, std::shared_ptr<HttpServer::Request> request) {

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

        if (service_pool_ == NULL) {
          LOG(ERROR) << "Engine pool not initailized. " << endl;
          return;
        }

        service_pool_->enqueue(&data);

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
private:
  SystemAppsService sys_apps_;

};
}

#endif //MATRIX_APPS_RESTFUL_H_
