/*
 * dog_common.h
 *
 *  Created on: Dec 21, 2015
 *      Author: chenzhen
 */

#ifndef DOG_COMMON_H_
#define DOG_COMMON_H_

const unsigned int FINGER_PRINT_LEN = 16;
const unsigned int AES_KEY_LEN = 32;
const unsigned int AES_IV_LEN = 16;

typedef struct {
     unsigned char cpuId[256];
     unsigned char biosUUID[256];
     unsigned char mac[256];
     unsigned char gpuId[1024];
} HardwareInfo;

void getCpuId(unsigned char *cpuId, int &len);
void getBiosUUID(unsigned char *uuid, int &len);
void getMac(unsigned char *mac, int &len);
void getGpuId(unsigned char *gpuId, int &len, int &gpuNum);
void getHardwareInfo(HardwareInfo &hardwareInfo);

int compare(const unsigned char *a, const int lenA, const unsigned char *b,
            const int lenB);
int fingerprint(HardwareInfo *info, unsigned char *figerprint);

int aesEncrypt(unsigned char *modelData, const unsigned long len,
               unsigned char *enModelData, unsigned char *modelKey);

int aesDecrypt(unsigned char *enModelData, const unsigned long enLen,
               unsigned char *modelData, unsigned char *modelKey);

#endif /* DOG_COMMON_H_ */
