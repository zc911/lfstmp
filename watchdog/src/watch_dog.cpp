/*
 * dog_client.c

 *
 *  Created on: Nov 27, 2015
 *      Author: chenzhen
 */

#include "watch_dog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <glog/logging.h>
#include <openssl/aes.h>
#include "common.h"
#include "dog.h"

using namespace std;

int CheckHardware() {

    unsigned char fpKey[FINGER_PRINT_LEN] = { 0 };

    KeyStatus status = readHwFingerprintFromDog(0, fpKey);
    if (status != KEY_OK) {
        LOG(ERROR)<< "Read key error." << endl;
        return ERR_DOG_INVALID;
    }

    unsigned char fp[FINGER_PRINT_LEN] = { 0 };
    HardwareInfo hardwareInfo;
    getHardwareInfo(hardwareInfo);
    fingerprint(&hardwareInfo, fp);

    if (compare(fp, sizeof(fp), fpKey, sizeof(fpKey)) != 0) {
        LOG(ERROR)<< "Invalid key" << endl;
        return ERR_INVALID_KEY;
    }

    memset(fpKey, 0, sizeof(fpKey));
    memset(fp, 0, sizeof(fp));

    return ERR_SUCCESS;

}

void* monitor(void *p) {
    while (1) {
        sleep(MONITOR_INTERVAL);
        if (loginDog(0) != KEY_OK) {
            LOG(ERROR)<< "Dog monitor error." << endl;
            exit(-1);
        } else {
            logoutDog(0);
        }

        DLOG(INFO)<< "Dog monitor..." << endl;
    }
    return NULL;
}
int DummyWatchDog(const unsigned char *msg){

     if (resetDogDataWithLog(0,msg) != KEY_OK) {
         return ERR_DOG_INVALID;
     }
     return ERR_SUCCESS;
}
int StartDogMonitor() {
    pthread_t id;
    return pthread_create(&id, NULL, monitor, NULL);
}

int EncryptModel(unsigned char *modelData, const unsigned long len,
                 unsigned char *enModelData) {

    unsigned char modelKey[AES_KEY_LEN];
    int keyLen;
    if (readModelKeyFromDog(0, modelKey, keyLen) != KEY_OK) {
        return ERR_DOG_INVALID;
    }

    return aesEncrypt(modelData, len, enModelData, modelKey) == 0 ?
            ERR_SUCCESS : ERR_ENCRYPT_SYS_ERROR;

}

int DecryptModel(unsigned char *enModelData, const unsigned long enLen,
                 unsigned char *modelData) {

    unsigned char modelKey[AES_KEY_LEN];
    int keyLen;
    if (readModelKeyFromDog(0, modelKey, keyLen) != KEY_OK) {
        return ERR_DOG_INVALID;
    }

    return aesDecrypt(enModelData, enLen, modelData, modelKey) == 0 ?
            ERR_SUCCESS : ERR_ENCRYPT_SYS_ERROR;

}

int EncryptModel(unsigned char *modelData, const unsigned long len,
                 unsigned char *enModelData, unsigned char *key) {
    unsigned char modelKey[AES_KEY_LEN];
    memcpy(modelKey, key, AES_KEY_LEN);

    return aesEncrypt(modelData, len, enModelData, modelKey) == 0 ?
            ERR_SUCCESS : ERR_ENCRYPT_SYS_ERROR;
}

int DecryptModel(unsigned char *enModelData, const unsigned long enLen,
                 unsigned char *modelData, unsigned char *key) {
    unsigned char modelKey[AES_KEY_LEN];
    memcpy(modelKey, key, AES_KEY_LEN);

    return aesDecrypt(enModelData, enLen, modelData, modelKey) == 0 ?
            ERR_SUCCESS : ERR_ENCRYPT_SYS_ERROR;
}

// Currently, AWS China has no KMS service.
// So we use a fake key instead of getting the key from KMS
#define AWS_KEY_FAKE "asdg17fhrg23fvvd63411rsdcv0vwet1"
int EncryptModelAWS(unsigned char *modelData, const unsigned long len,
                    unsigned char *enModelData) {
    unsigned char awsKey[AES_KEY_LEN];
    memcpy(awsKey, AWS_KEY_FAKE, AES_KEY_LEN);
    return EncryptModel(modelData, len, enModelData, awsKey);
}

int DecryptModelAWS(unsigned char *enModelData, const unsigned long enLen,
                    unsigned char *modelData) {
    unsigned char awsKey[AES_KEY_LEN];
    memcpy(awsKey, AWS_KEY_FAKE, AES_KEY_LEN);
    return DecryptModel(enModelData, enLen, modelData, awsKey);
}

int CheckFeature(Feature f, FeatureSet fs) {
    if ((f & fs) > 0) {
        return ERR_FEATURE_ON;
    }
    return ERR_FEATURE_OFF;
}

int GetFeatureSet(FeatureSet &fs) {

    if (readFeaturesFromDog(0, fs) == KEY_OK) {
        return ERR_SUCCESS;
    } else {
        return ERR_FEATURE_ERROR;
    }
}

int CheckFeature(Feature f) {
    FeatureSet fs;
    readFeaturesFromDog(0, fs);
    return CheckFeature(f, fs);
}

