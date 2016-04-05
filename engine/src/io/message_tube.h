/*
 * message_tube.h
 *
 *  Created on: 11/03/2016
 *      Author: chenzhen
 */

#ifndef MESSAGE_TUBE_H_
#define MESSAGE_TUBE_H_

#include <pthread.h>
#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "model/model.h"
#include "network/data_service_client.h"

namespace deepglint {

class MessageTube {
 public:
    MessageTube(const string addr, unsigned int queueSize,
                unsigned int batchSize);
    virtual ~MessageTube();

    void InsertQueue(Frame *frame);
    int StartAsyn();

 private:
    void send();
    void sendToServer(vector<Message> &vehicleMsg);

 private:
    boost::circular_buffer<Message> *content_;
    boost::mutex mutex_;
    pthread_t tid_;
    unsigned int buffer_size_;
    unsigned int cur_send_index_;
    unsigned int batch_size_;
    VideoMetaData video_meta_data_;
    DataServiceClient *client_;

};

}
#endif /* MESSAGE_TUBE_H_ */
