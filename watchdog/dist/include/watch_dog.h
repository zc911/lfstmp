/*
 * dog_client.h
 *
 *  Created on: Nov 26, 2015
 *      Author: chenzhen
 */

#ifndef SRC_INCLUDE_DOG_CLIENT_H_
#define SRC_INCLUDE_DOG_CLIENT_H_

#include <stdint.h>

#define ERR_SUCCESS 0x00
#define ERR_INVALID_MODEL 0x01
#define ERR_INVALID_KEY 0x02
#define ERR_EXE_KEY 0x03
#define ERR_INVALID_PARAM 0x0
#define ERR_SYS_INFO 0x11
#define ERR_DOG_NOT_FOUND 0x21
#define ERR_DOG_INVALID 0x22
#define ERR_DOG_PIN_ERROR 0x23
#define ERR_ENCRYPT_INVALID_PARAM 0x31
#define ERR_ENCRYPT_SYS_ERROR 0x32

#define ERR_FEATURE_ERROR 0x40
#define ERR_FEATURE_ON 0x41
#define ERR_FEATURE_OFF 0x42

#define MONITOR_INTERVAL 60
#define CHECK_INTERVAL 30

using namespace std;

enum Feature {
    FEATURE_NONO = 0,
    FEATURE_CAR_DETECTION = 1,
    FEATURE_RESERVED = 2,
    FEATURE_CAR_STYLE = 4,
    FEATURE_CAR_COLOR = 8,
    FEATURE_CAR_PLATE = 16,
    FEATURE_CAR_MARKER = 32,
    FEATURE_CAR_RANKER = 64,
    FEATURE_CAR_EXTRACTOR=128,
    FEATURE_FACE_EXTRACTOR=256,
    FEATURE_FACE_RANKER=512

};

typedef uint32_t FeatureSet;

/**
 * This function check the hardware info and compare it
 * with the info stored within dog.
 * return ERR_SUCCESS if check successful.
 * return error message if check failed.
 */
int CheckHardware();

/**
 * This function start a monitor thread to
 * check whether the dog is plugged in or not.
 * If not, it will call exit(-1) to terminate
 * the current process.
 * The interval in default is MONITOR_INTERVAL(60) seconds.
 */
int StartDogMonitor();

/**
 * Get feature set from dog.
 * A feature set is an 32 bit unsigned int type,
 * and each bit represents whether the corresponding features
 * ON or OFF.
 * Use the CheckFeature function to get a friendly result
 */
int GetFeatureSet(FeatureSet &fs);

/**
 * Check the Feature f is ON or NOT by FeatureSet fs.
 */
int CheckFeature(Feature f, FeatureSet fs);

/**
 * Check Feature f is ON or OFF.
 * Call of this function will lead to data reading from dog.
 */
int CheckFeature(Feature f,bool isRunning);

/**
 * This function will encrypt data using key stored within dog.
 */
int EncryptModel(unsigned char *modelData, const unsigned long len,
                 unsigned char *enModelData);

/**
 * This function will decrypt data using key stored within dog.
 */
int DecryptModel(unsigned char *enModelData, const unsigned long enLen,
                 unsigned char *modelData);

/**
 * This function will encrypt data using key provided from param key.
 * The length of the key muse be 32 bits, otherwise error occurs.
 */
int EncryptModel(unsigned char *modelData, const unsigned long len,
                 unsigned char *enModelData, unsigned char *key);

/**
 * This function will decrypt data using key provided from param key.
 * The length of the key muse be 32 bits
 */
int DecryptModel(unsigned char *enModelData, const unsigned long enLen,
                 unsigned char *modelData, unsigned char *key);

/**
 * This function will encrypt data using key stored in AWS KMS
 */
int EncryptModelAWS(unsigned char *modelData, const unsigned long len,
                    unsigned char *enModelData);

/**
 * This function will decrypt data using key stored in AWS KMS
 */
int DecryptModelAWS(unsigned char *enModelData, const unsigned long enLen,
                    unsigned char *modelData);

#endif /* SRC_INCLUDE_DOG_CLIENT_H_ */
