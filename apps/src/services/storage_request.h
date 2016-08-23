//
// Created by jiajaichen on 16-6-15.
//

#ifndef PROJECT_STORAGE_REQUEST_H
#define PROJECT_STORAGE_REQUEST_H
#include "codec/base64.h"
#include "witness_bucket.h"
#include "clients/spring_client.h"
#include "clients/data_client.h"
#include "simple_thread_pool.h"

namespace dg {
class StorageRequest {
public:
    StorageRequest(const Config *config) {
        string address = (string) config->Value(STORAGE_ADDRESS);
        spring_client_.CreateConnect(address);
        data_client_.CreateConnect(address);
        pool_ = new ThreadPool(4);
    }

    MatrixError storage() {
        VLOG(VLOG_SERVICE) << "========START REQUEST "<<WitnessBucket::Instance().Size()<<"===========" << endl;

        MatrixError err;
        shared_ptr<WitnessVehicleObj> wv = WitnessBucket::Instance().Pop();

        for (int k = 0; k < wv->results.size(); k++) {
            VehicleObj vo;
            PedestrianObj po;
            const WitnessResult &r = wv->results.Get(k);
            for (int i = 0; i < r.vehicles_size(); i++) {
                Cutboard c = r.vehicles(i).img().cutboard();
                Mat roi(wv->imgs[k], Rect(c.x(), c.y(), c.width(), c.height()));
                RecVehicle *v = wv->results.Mutable(k)->mutable_vehicles(i);
                vector<uchar> data;
                imencode(".jpg",roi,data);

             //   vector<uchar> data(roi.datastart, roi.dataend);
                string imgdata = Base64::Encode(data);
                v->mutable_img()->mutable_img()->set_bindata(imgdata);
                vo.mutable_vehicle()->Add()->CopyFrom(*v);


            }
            for (int i = 0; i < r.pedestrian_size(); i++) {
                Cutboard c = r.pedestrian(i).img().cutboard();
                Mat roi(wv->imgs[k], Rect(c.x(), c.y(), c.width(), c.height()));
                RecPedestrian *v = wv->results.Mutable(k)->mutable_pedestrian(i);
                po.mutable_pedestrian()->Add()->CopyFrom(*v);
                                vector<uchar> data;

                imencode(".jpg",roi,data);
            //    vector<uchar> data(roi.datastart, roi.dataend);
                string imgdata = Base64::Encode(data);
                v->mutable_img()->mutable_img()->set_bindata(imgdata);
                po.mutable_pedestrian()->Add()->CopyFrom(*v);
            }
            if (r.vehicles_size() > 0) {
                vo.mutable_metadata()->CopyFrom(wv->srcMetadatas[k]);
                vo.mutable_img()->set_uri(r.image().data().uri());
                vo.mutable_img()->set_height(wv->imgs[k].rows);
                vo.mutable_img()->set_width(wv->imgs[k].cols);
            }
            if (r.pedestrian_size()) {
                po.mutable_metadata()->CopyFrom(wv->srcMetadatas[k]);
                po.mutable_img()->set_uri(r.image().data().uri());
                po.mutable_img()->set_height(wv->imgs[k].rows);
                po.mutable_img()->set_width(wv->imgs[k].cols);
            }


            for (int i = 0; i < wv->storages.size(); i++) {

                string address = wv->storages.Get(i).address();
                if (wv->storages.Get(i).type() == model::POSTGRES) {
                    pool_->enqueue([address, this, vo, po, &err]() {
                        MatrixError errTmp = this->data_client_.SendBatchData(address, vo, po);
                        if (errTmp.code() != 0) {
                            err.set_code(errTmp.code());
                            err.set_message("send to postgres error");
                        }
                    });
                } else if (wv->storages.Get(i).type() == model::KAFKA) {
                    pool_->enqueue([address, this, vo, &err]() {
                        MatrixError errTmp = spring_client_.IndexVehicle(address, vo);
                        if (errTmp.code() != 0) {
                            err.set_code(errTmp.code());
                            err.set_message("send to kafka error");
                        }
                    });

                }
            }
        }
        return err;
    }
    ~StorageRequest() {
        if (pool_) {
            delete pool_;
        }
    }
private:
    ThreadPool *pool_;
    SpringClient spring_client_;
    DataClient data_client_;
};
}
#endif //PROJECT_STORAGE_REQUEST_H
