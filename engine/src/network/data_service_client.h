/*
 * data_service.h
 *
 *  Created on: 14/03/2016
 *      Author: chenzhen
 */

#ifndef DATA_SERVICE_H_
#define DATA_SERVICE_H_
#include "model/model.h"
namespace dg {
/**
 * The data service client interface
 */
class DataServiceClient {
public:
    DataServiceClient(const string addr)
        : addr_(addr) {

    }

    virtual ~DataServiceClient() {

    }

    virtual void SendData(Message &msg) = 0;
    virtual void SendBatchData(vector<Message> &msg) = 0;

protected:
    string addr_;
};
}
#endif /* DATA_SERVICE_H_ */
