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

#include "pbjson.hpp" //from pbjson
#include "server_http.hpp" //from Simple-Web-Server


namespace dg
{

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;

class RestfulService
{
public:
    RestfulService(){}
    virtual ~RestfulService(){}

    virtual void Bind(HttpServer& server) = 0;
};

template<class request_type, class response_type>
class RestfulBinder
{
public:
    RestfulBinder(std::function<bool(const request_type*, response_type*)> func)
            : func_(func)
    {

    }
    virtual ~RestfulBinder(){}

    void Bind(HttpServer& server, string endpoint, string method)
    {
        server.resource[endpoint][method] = [this](HttpServer::Response& response, std::shared_ptr<HttpServer::Request> request)
        {
            request_type protobufRequestMessage;
            response_type protobufResponseMessage;

            try
            {
                string content = request->content.string();
                string err;
                int ret = pbjson::json2pb(content, &protobufRequestMessage, err);
                if (ret < 0)
                {
                    responseText(response, 400, "parameter conversion failed: " + err);
                    return;
                }

                if (!func_(&protobufRequestMessage, &protobufResponseMessage))
                {
                    responseText(response, 500, "call method failed");
                    return;
                }

                content = "";
                pbjson::pb2json(&protobufResponseMessage, content);
                responseText(response, 200, content);
            }
            catch (exception& e)
            {
                responseText(response, 500, e.what());
            }
        };
    }

private:
    std::function<bool(const request_type*, response_type*)> func_;

    static void responseText(HttpServer::Response& response, int code, const string& text)
    {
        response << "HTTP/1.1 " << std::to_string(code) << "\r\nContent-Length: "
                 << text.length() << "\r\n\r\n" << text;
    }
};

}

#endif //MATRIX_APPS_RESTFUL_H_
