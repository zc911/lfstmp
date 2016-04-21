/*
 * basic_service.h
 *
 *  Created on: Apr 21, 2016
 *      Author: chenzhen
 */

#ifndef BASIC_SERVICE_H_
#define BASIC_SERVICE_H_

namespace dg {
class BasicService {
 public:
    virtual ~BasicService() {

    }
    virtual void Run() = 0;
};
}

#endif /* BASIC_SERVICE_H_ */
