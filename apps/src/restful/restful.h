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
#include "http_server.hpp" //from Simple-Web-Server


namespace dg
{


class RestfulService
{
public:
    template <class socket_type>
    virtual void Bind(SimpleWeb::ServerBase<socket_type>& server);

protected:
    template<class socket_type, class request_type, class response_type>
    void bind(SimpleWeb::ServerBase<socket_type>& server, string endpoint, string method, bool caller(const request_type* req, response_type* resp))
    {
        server.resource[endpoint][method] = [](SimpleWeb::ServerBase<socket_type>::Response& response, shared_ptr<SimpleWeb::ServerBase<socket_type>::Request> request)
        {
            request_type protobufRequestMessage;
            response_type protobufResponseMessage;

            try 
            {
                string content = request->content.string();
                string err;
                int ret = pbjson::json2pb(content, &protobufRequestMessage, err)
                if (ret < 0)
                {
                    responseText(response, 400, "parameter conversion failed: " + err);
                    return
                }

                if (!caller(&protobufRequestMessage, &protobufResponseMessage))
                {
                    responseText(response, 500, "call method failed");
                    return;
                }

                pbjson::pb2json(&protobufResponseMessage, content);
                responseText(response, 200, content);
            } 
            catch (exception& e) 
            {
                responseText(response, 500, e.what());
            }
        }
    }

private:
    template <class socket_type>
    void responseText(SimpleWeb::ServerBase<socket_type>::Response& response, int code, string& text)
    {
        response << "HTTP/1.1 " << code << "\r\nContent-Length: " 
                 << text.length() << "\r\n\r\n" << text;
    }
};
}

#endif //MATRIX_APPS_RESTFUL_H_