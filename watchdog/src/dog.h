/*
 * key.h
 *
 *  Created on: Dec 22, 2015
 *      Author: chenzhen
 */

#ifndef KEY_H_
#define KEY_H_

#include <stdint.h>
#include "common.h"
#include "utils/tool.h"

enum KeyStatus {
    KEY_OK = 0,
    KEY_NOT_FOUND = 100,
    KEY_PW_INVALID = 101,
    KEY_ERROR = 200,
    KEY_SYS_ERROR = 201
};


typedef struct {
    unsigned char version[4];
    unsigned char capacity[4];
    unsigned char features[4];
    unsigned char reserved[8];
    unsigned char hardwareInfo[FINGER_PRINT_LEN];
    unsigned char modelKey[AES_KEY_LEN];

} DogData;


extern void printHex(void *data, const int len);
KeyStatus loginDog(const int dogId);
KeyStatus logoutDog(const int dogId);
KeyStatus readDog(const int dogId, const int offset, unsigned char *dest,
                  int len);
KeyStatus writeDog(const int dogId, const int offset, const unsigned char *src,
                   int len);
KeyStatus resetDogData(const int dogId);
KeyStatus resetDogDataWithLog(const int dogId,const unsigned char *msg);
KeyStatus readReservedFromDog(const int dogId, uint32_t &code, uint32_t &time);
KeyStatus encryptByDog(const int dogId, unsigned char *data, int len);
KeyStatus decryptByDog(const int dogId, unsigned char *data, int len);
KeyStatus checkDogStatus(const int dogId);
KeyStatus readDogData(const int dogId, DogData &data);
KeyStatus writeDogData(const int dogId, DogData &data);
KeyStatus writeHwInfoIntoDog(const int dogId, HardwareInfo *info);
KeyStatus readHwFingerprintFromDog(const int dogId, unsigned char *fingerprint);
KeyStatus writeModelKeyIntoDog(const int dogId, const unsigned char *key,
                               const int len);
KeyStatus readModelKeyFromDog(const int dogId, unsigned char *key, int &len);

KeyStatus readFeaturesFromDog(const int dogId, uint32_t &feature);
KeyStatus writeFeaturesIntoDog(const int dogId, const uint32_t &feature);

#endif /* KEY_H_ */
