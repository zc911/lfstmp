//
// Created by jiajaichen on 16-6-15.
//

#ifndef PROJECT_STORAGE_REQUEST_H
#define PROJECT_STORAGE_REQUEST_H
#include "witness_bucket.h"
#include "clients/spring_client.h"
#include "clients/data_client.h"

namespace dg {
class StorageRequest {
public:
    StorageRequest(const Config *config) {
        string address = (string) config->Value(STORAGE_ADDRESS);
        spring_client_.CreateConnect(address);
        data_client_.CreateConnect(address);

    }

    MatrixError storage() {
        VLOG(VLOG_SERVICE) << "========START REQUEST===========" << endl;

        MatrixError err;
        shared_ptr<WitnessVehicleObj> wv = WitnessBucket::Instance().Pop();

        string address = wv->storage().address();
        for(int i=0;i<=wv->storage().types_size();i++){
            if(wv->storage().types(i)==model::POSTGRES){
                MatrixError errTmp=data_client_.SendBatchData(address,wv->mutable_vehicleresult());
                if(errTmp.code()!=0){
                    err.set_code(errTmp.code());
                    err.set_message("send to postgres error");
                }
            }else if(wv->storage().types(i)==model::KAFKA){
                const VehicleObj &v = wv->vehicleresult();
                MatrixError errTmp=spring_client_.IndexVehicle(address,v);
                if(errTmp.code()!=0){
                    err.set_code(errTmp.code());
                    err.set_message("send to kafka error");
                }
            }
        }
        return err;
    }
    ~StorageRequest() { }
private:
    SpringClient spring_client_;
    DataClient data_client_;
};
}
#endif //PROJECT_STORAGE_REQUEST_H
