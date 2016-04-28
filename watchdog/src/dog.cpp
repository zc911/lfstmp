#include "dog.h"
#include <iostream>
#include <glog/logging.h>

using namespace std;

KeyStatus resetDogData(const int dogId) {
    DogData data;
    memset(&data, 0, sizeof(data));
    return writeDogData(dogId, data);
}

KeyStatus writeHwInfoIntoDog(const int dogId, HardwareInfo *info) {

    unsigned char fp[FINGER_PRINT_LEN] = { 0 };
    fingerprint(info, fp);
    DogData data;
    KeyStatus status = readDogData(dogId, data);
    if (status != KEY_OK) {
        return status;
    }
    memcpy(data.hardwareInfo, fp, sizeof(fp));
    status = writeDogData(dogId, data);
    if (status != KEY_OK) {
        return status;
    }

    return KEY_OK;
}
KeyStatus readReservedFromDog(const int dogId, uint32_t &code, uint32_t &time){
     DogData data;
     KeyStatus status = readDogData(dogId,data);
     if(status!=KEY_OK){
          return status;
     }
     uint32_t c,t;
     memcpy(&c,data.reserved,4);
     memcpy(&t,data.reserved+4,4);
     code = c;
     time = t;
     return status;

}
KeyStatus readHwFingerprintFromDog(const int dogId,
                                   unsigned char *fingerprint) {

    DogData data;
    KeyStatus status = readDogData(dogId, data);
    if (status != KEY_OK) {
        return status;
    }
    memcpy(fingerprint, data.hardwareInfo, sizeof(data.hardwareInfo));
    return KEY_OK;
}

KeyStatus writeModelKeyIntoDog(const int dogId, const unsigned char *key,
                               const int len) {

    KeyStatus status = loginDog(dogId);
    if (status != KEY_OK) {
        return status;
    }
    unsigned char keyEn[AES_KEY_LEN];
    memcpy(keyEn, key, AES_KEY_LEN);
    status = encryptByDog(dogId, keyEn, AES_KEY_LEN);

    if (status != KEY_OK) {
        return status;
    }

    DogData data;
    status = readDogData(dogId, data);
    if (status != KEY_OK) {
        return status;
    }

    memcpy(data.modelKey, keyEn, AES_KEY_LEN);
    status = writeDogData(dogId, data);
    memset(&data, 0, AES_KEY_LEN);
    memset(keyEn, 0, AES_KEY_LEN);
    if (status != KEY_OK) {
        return status;
    }

    return KEY_OK;
}

KeyStatus readModelKeyFromDog(const int dogId, unsigned char *key, int &len) {
    DogData data;
    KeyStatus status = readDogData(dogId, data);
    if (status != KEY_OK) {
        return status;
    }

    status = loginDog(dogId);
    if (status != KEY_OK) {
        return status;
    }
    unsigned char keyDe[AES_KEY_LEN];
    memcpy(keyDe, data.modelKey, AES_KEY_LEN);
    memset(&data, 0, sizeof(data));
    status = decryptByDog(dogId, keyDe, AES_KEY_LEN);
    if (status != KEY_OK) {
        return status;
    }
    logoutDog(dogId);
    memcpy(key, keyDe, AES_KEY_LEN);
    memset(keyDe, 0, AES_KEY_LEN);
//    if (status != KEY_OK) {
//        return status;
//    }

    return KEY_OK;
}

KeyStatus readDogData(const int dogId, DogData &data) {
    memset(&data, 0, sizeof(data));
    KeyStatus status = loginDog(dogId);
    if (status != KEY_OK) {
        LOG(ERROR)<< "Key error " << status << endl;
        return status;
    }
    status = readDog(dogId, 0, (unsigned char *) &data, sizeof(data));
    if (status != KEY_OK) {
        LOG(ERROR)<< "Key IO error " << status << endl;
        logoutDog(dogId);
        return status;
    }
    logoutDog(dogId);
//    if (status != KEY_OK) {
//        LOG(ERROR)<< "Key error " << status << endl;
//    }
    return status;
}

KeyStatus writeDogData(const int dogId, DogData &data) {
    KeyStatus status = loginDog(dogId);
    if (status != KEY_OK) {
        LOG(ERROR)<< "Key error" << status << endl;
        return status;
    }

    status = writeDog(dogId, 0, (unsigned char *) &data, sizeof(data));
    if (status != KEY_OK) {
        LOG(ERROR)<< "Key IO error" << status << endl;
        logoutDog(dogId);
        return status;
    }
    logoutDog(dogId);
//    if (status != KEY_OK) {
//        LOG(ERROR)<< "Key error" << status << endl;
//    }
    return status;
}

KeyStatus readFeaturesFromDog(const int dogId, uint32_t &feature) {
    DogData data;
    KeyStatus status = readDogData(dogId, data);
    if (status != KEY_OK) {
        return status;
    }

    uint32_t f;
    memcpy(&f, data.features, 4);
    memset(&data, 0, sizeof(data));
    feature = f;
    return status;
}

KeyStatus writeFeaturesIntoDog(const int dogId, const uint32_t &feature) {

    DogData data;
    KeyStatus status = readDogData(dogId, data);
    if (status != KEY_OK) {
        return status;
    }

    memcpy(data.features, &feature, 4);
    status = writeDogData(dogId, data);
    memset(&data, 0, AES_KEY_LEN);

    return status;
}

KeyStatus resetDogDataWithLog(const int dogId,const unsigned char *msg){
     DogData data;
     memset(&data, 0, sizeof(data));
     memcpy(data.reserved,msg,8);

     return writeDogData(dogId, data);
}

