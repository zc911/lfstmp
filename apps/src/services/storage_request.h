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
        string storageAddress = (string) config->Value(STORAGE_ADDRESS);
        spring_client_.CreateConnect(storageAddress);
    }

    MatrixError storage() {
        VLOG(VLOG_SERVICE) << "========START REQUEST===========" << endl;

        MatrixError err;
        shared_ptr<WitnessVehicleObj> wv = WitnessBucket::Instance().Pop();

        string storageAddress = wv->storage().address();


        const VehicleObj &v = wv->vehicleresult();
        err=spring_client_.IndexVehicle(storageAddress,v);

        return err;
    }
    ~StorageRequest() { }
private:
    SpringClient spring_client_;
    DataClient data_client_;
};
}
#endif //PROJECT_STORAGE_REQUEST_H
