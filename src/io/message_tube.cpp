#include "message_tube.h"
#include <jsoncpp/json/json.h>
#include "network/data_service_grpc_client.h"
#include "util/string_util.h"

MessageTube::MessageTube(VideoMetaData &metaData, const string addr,
                         unsigned int queueSize, unsigned int batchSize) {
    buffer_size_ = queueSize;
    video_meta_data_ = metaData;
    content_ = new boost::circular_buffer<Message>(buffer_size_);
    client_ = new DataServiceGrpcClient(addr);
    cur_send_index_ = 0;
    batch_size_ = batchSize;
    tid_ = NULL;
}

MessageTube::~MessageTube() {
    delete content_;
}

void MessageTube::InsertQueue(Frame *frame) {
    vector<Vehicle> vs = frame->Vehicles();
    mutex_.lock();
    for (auto v : vs) {
        if (!v.vehicle_result.isClassified
                || !v.vehicle_result.isClassifiedFirst) {
            continue;
        }

        if (v.vehicle_result.confidence <= MODEL_TYPE_CONFIDENCE_THRESHOLDS) {
            continue;
        }
/*
        char name[1024];
        sprintf(name, "snapshots/%lld_%s_%s_%s_%f_%lld.jpg", v.id,
                v.vehicle_result.mainBrandName.c_str(),
                v.vehicle_result.subBrandName.c_str(),
                v.vehicle_result.modelYear.c_str(), v.vehicle_result.confidence,
                frame->Timestamp());

        SaveMatToFile(string(name), v.vehicle_pic.pic);
*/
        Message msg;
        msg.videoMetaData = video_meta_data_;
        msg.timestamp = frame->Timestamp();
        msg.status = MESSAGE_STATUS_INIT;
        msg.vehicle = v;
        content_->push_back(msg);

    }
    mutex_.unlock();

}
void MessageTube::sendToServer(vector<Message> &vehicleMsg) {
    if (vehicleMsg.size() > 0)
        client_->SendBatchData(vehicleMsg);
}

void MessageTube::send() {
    vector<Message> vehicleMsg(batch_size_);
    for (;;) {
        vehicleMsg.clear();
        mutex_.lock();
        for (int i = 0; i < batch_size_; ++i) {
            ++cur_send_index_;
            if (cur_send_index_ >= content_->size()) {
                cur_send_index_ = 0;
            }
            if (cur_send_index_ >= content_->size()) {
                break;
            }
            Message m = (*content_)[cur_send_index_];
            if (m.status == MESSAGE_STATUS_SENT) {
                DLOG(WARNING)<< "OLD MESSAGE" << endl;
                break;
            }

            vehicleMsg.push_back(m);
            ((*content_)[cur_send_index_]).status = MESSAGE_STATUS_SENT;

        }
        mutex_.unlock();
        sendToServer(vehicleMsg);
        usleep(500 * 1000);
    }

}

int MessageTube::StartAsyn() {
    typedef void* (*FUNC)(void*);
    FUNC callback = (FUNC) &MessageTube::send;
    pthread_create(&tid_, NULL, callback, (void*) this);
    return 0;
}
