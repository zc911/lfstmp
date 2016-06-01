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
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

#include "pbjson/pbjson.hpp" //from pbjson
#include "Simple-Web-Server/server_http.hpp" //from Simple-Web-Server
#include "../model/common.pb.h"
#include "debug_util.h"

using namespace std;
using namespace ::dg::model;
namespace dg {

template<class request_type, class response_type>
using BindFunction = std::function<MatrixError(const request_type *, response_type *)>;

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;

//template<class request_type, class response_type>
class RestfulService {
public:
    RestfulService(string protocol = "HTTP/1.1", string mime_type =
    "application/json; charset=utf-8")
        : protocol_(protocol),
          mime_type_(mime_type) {
    }

    virtual ~RestfulService() {
    }

//    template<class request_type, class response_type>
    class CallData {
    public:
        bool Wait() {
            std::unique_lock<std::mutex> lock(m);
            cond.wait(lock);
            return true;
        }
        void Finish() {
            cond.notify_all();
        }
        void operator()(WitnessAppsService *apps) {
//            std::function<MatrixError()> f = std::bind(func, apps);
//            f(apps);
            Finish();
        }
    private:
        friend class RestfulService;
        WitnessAppsService *apps;
        std::function<MatrixError()> func;
        std::mutex m;
        std::condition_variable cond;

    };

//    template<class request_type, class response_type>
    void StartThread(WitnessAppsService *apps) {

        workers_.emplace_back([this, apps] {
          for (; ;) {
              CallData *task;
              {

                  std::unique_lock<std::mutex> lock(queue_mutex_);
                  condition_.wait(lock, [this] {
                    return (this->stop_ || !this->tasks_.empty());
                  });

                  if (this->stop_ || this->tasks_.empty())
                      return;

                  task = this->tasks_.front();
                  this->tasks_.pop();
                  lock.unlock();
              }
              task->apps = apps;
              MatrixError err = task->func();
//              task->Finish();
//              (*task)();
              cout << "finish batch rec: " << endl;
          }
        });

    }


//    template<class F, ponse_type>
    auto enqueue(CallData *data) {


        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                cout << "is stop" << endl;
                return 1;
            }
            tasks_.push(data);
        }
        condition_.notify_one();
        return 1;

    };

    virtual void Bind(HttpServer &server) = 0;

protected:
    queue<CallData *> tasks_;
    vector<std::thread> workers_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;


    template<class apps_type, class request_type, class response_type>
    void bindFunc(HttpServer &server,
                  string endpoint,
                  string method,
                  MatrixError(*func)(const request_type *, response_type *)
    ) {

//              std::function<MatrixError(const request_type *, response_type *)> func) {

        server.resource[endpoint][method] =
            [this, func](HttpServer::Response &response, std::shared_ptr<HttpServer::Request> request) {
              cout << "========================" << endl;
              cout << "Get request, thread id: " << this_thread::get_id() << endl;
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
//                  MatrixError error = (&protobufRequestMessage, &protobufResponseMessage);
                  CallData data;
                  typedef MatrixError(apps_type::*MemFunc)(const request_type *, response_type *);
                  MemFunc memFunc = (MemFunc) func;
                  data.func = [memFunc, &protobufRequestMessage, &protobufResponseMessage, &data]() -> MatrixError {

                    return (bind(memFunc, data.apps, placeholders::_1, placeholders::_2))(&protobufRequestMessage,
                                                                                       &protobufResponseMessage);
                  };
//                  data.req = &protobufRequestMessage;
//                  data.resp = &protobufResponseMessage;
//                  this->tasks_.emplace();
                  this->enqueue(&data);

                  if (data.Wait()) {
                      cout << "Finish" << endl;
                  }

//                  if (error.code() != 0) {
//                      responseText(response, 500, error.message());
//                      return;
//                  }

                  content = "";
                  pbjson::pb2json(&protobufResponseMessage, content);
                  responseText(response, 200, content);
                  gettimeofday(&end, NULL);
                  cout << "Request cost: " << TimeCostInMs(start, end) << endl;
              }
              catch (exception &e) {
                  responseText(response, 500, e.what());
              }
            };
    }


//    template<class request_type, class response_type>

//    static void bind(
//        HttpServer &server, string endpoint, string method,
//        std::function<MatrixError(const request_type *, response_type *)> func) {
//
//        server.resource[endpoint][method] =
//            [func](HttpServer::Response &response, std::shared_ptr<HttpServer::Request> request) {
//
//              request_type protobufRequestMessage;
//              response_type protobufResponseMessage;
//
//              try {
//                  string content = request->content.string();
//                  string err;
//                  int ret = pbjson::json2pb(content, &protobufRequestMessage, err);
//                  if (ret < 0) {
//                      responseText(response, 400, "parameter conversion failed: " + err);
//                      return;
//                  }
//                  MatrixError error = func(&protobufRequestMessage, &protobufResponseMessage);
//                  if (error.code() != 0) {
//                      responseText(response, 500, error.message());
//                      return;
//                  }
//
//                  content = "";
//                  pbjson::pb2json(&protobufResponseMessage, content);
//
//                  responseText(response, 200, content);
//
//              }
//              catch (exception &e) {
//                  responseText(response, 500, e.what());
//              }
//            };
//    }

private:
    string protocol_;
    string mime_type_;

    static void responseText(HttpServer::Response &response, int code,
                             const string &text) {
        response << "HTTP/1.1 " << std::to_string(code)
            << "\r\nContent-Length: " << text.length()
            << "\r\nContent-Type: application/json; charset=utf-8\r\n\r\n"
            << text;
    }
};

}

#endif //MATRIX_APPS_RESTFUL_H_
