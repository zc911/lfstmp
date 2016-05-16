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
#include "pbjson/pbjson.hpp" //from pbjson
#include "Simple-Web-Server/server_http.hpp" //from Simple-Web-Server
#include "../model/common.pb.h"

using namespace std;
using namespace ::dg::model;
namespace dg {

template<class request_type, class response_type>
using BindFunction = std::function<MatrixError(const request_type *, response_type *)>;

typedef SimpleWeb::Server <SimpleWeb::HTTP> HttpServer;

class RestfulService {
public:
    RestfulService(string protocol = "HTTP/1.1", string mime_type =
    "application/json; charset=utf-8")
        : protocol_(protocol),
          mime_type_(mime_type) {
    }
    virtual ~RestfulService() {
    }

    virtual void Bind(HttpServer &server) = 0;

protected:
    template<class request_type, class response_type>
    static void bind(
        HttpServer &server, string endpoint, string method,
        std::function<MatrixError(const request_type *, response_type *)> func) {
        server.resource[endpoint][method] =
            [func](HttpServer::Response &response, std::shared_ptr<HttpServer::Request> request) {
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
                      responseText(response, 500, error.message());
                      return;
                  }

                  content = "";
                  pbjson::pb2json(&protobufResponseMessage, content);
                  responseText(response, 200, content);
              }
              catch (exception &e) {
                  responseText(response, 500, e.what());
              }
            };
    }

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
