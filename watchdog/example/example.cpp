/*
 * sys_get.c
 *
 *  Created on: Nov 29, 2015
 *      Author: chenzhen
 */

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <glog/logging.h>

#include "watch_dog.h"

using namespace std;

void printH(void *data, const int len, const char *title) {
    printf("\n-------%s-------\n", title);
    for (int i = 0; i < len; ++i) {
        printf("%02x-", ((unsigned char *) data)[i]);
    }
    printf("\n-------%s-------\n", title);
}

//void printInfo(HardwareInfo info) {
//    printf("----HW INFO----\n");
//    printf("CPU: %s \n", info.cpuId);
//    printf("MAC: %s \n", info.mac);
//    printf("BIOS: %s \n", info.biosUUID);
//    printf("GPU: %s \n", info.gpuId);
//    printf("----HW INFO----\n");
//}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
//    resetDogData(0);
//    unsigned char key[32];
//    unsigned char readKey[32];
//    generateKey(key, 32);
//    printH(key, 32, "Original Key");
//
//    writeModelKeyIntoDog(0, key, 32);
//    int n;
//    readModelKeyFromDog(0, readKey, n);
////
//    printH(readKey, 32, "Decrypt Key from dog");

//    HardwareInfo info;
//    getHardwareInfo(info);
//    writeHwInfoIntoDog(0, &info);
//
    unsigned char data[1023] = { 1 };
    unsigned char enData[1023] = { 0 };
    unsigned char deData[1023] = { 0 };

    for (int i = 0; i < 1023; ++i) {
        data[i] = (unsigned char) rand();
    }

    printH(data, 10, "data");
    printH(data + 1000, 24, "data");

    EncryptModel(data, 1023, enData);
    printH(enData, 10, "En data");

    DecryptModel(enData, 1023, deData);
    printH(deData, 10, "De data");
    printH(deData + 1000, 24, "De data");

    if (CheckHardware() != ERR_SUCCESS) {
        printf("Check hardware error \n");
        return -1;
    }

    printf("Start dog monitor...\n");
    StartDogMonitor();

    while (1) {
        sleep(100000);
    }

//
//    for (int i = 0; i < 1023; ++i) {
//        if (data[i] != deData[i]) {
//            printf("Find error: %d: %02x -> %02x \n", i, data[i], deData[i]);
//        }
//    }

//    HardwareInfo info;
//    getHardwareInfo(info);
//    printInfo(info);
//    fingerprint(&info, NULL);
//    writeHwInfoIntoDog(0, &info);
//    unsigned char fp[FINGER_PRINT_LEN];
//    readHwFingerprintFromKey(0, fp);

//    if (CheckHardware() != ERR_SUCCES) {
//        LOG(INFO)<< "Check hardware error: " << endl;
//    } else {
//        LOG(INFO) << "Check hardware succussful" << endl;
//    }
//
//    unsigned char key[32];
//    int len;
//    generateKey(key, KEY_LEN_DEFAULT);
//    DogData data;
//    readDogData(0, data);

//    char gpuId[1024];
//    int gpuNum;
//    int len;
//    getGpuId((unsigned char*) gpuId, len, gpuNum);
//    gpuId[len] = '\0';
//
//    printf("GPU info: %s \n", gpuId);
//    printf("Len: %d \n", len);
//    printf("Gpu num %d \n", gpuNum);

//  printf("0\n");
//  unsigned char key[32];
//  strncpy((unsigned char *) key, "RFbD56TI2smTyVsGd5Xav0yu99ZAMPTA", 32);
//
//  unsigned char iv[16];
//  strncpy((unsigned char *) iv, "RFbD56TI2smTyVsG", 16);
//
//  if (argc >= 2) {
//      unsigned char hardwareInfo[256];
//      unsigned char hipher[256];
//      unsigned char content[256];
//      char filePath[256];
//      strcpy(filePath, argv[1]);
//      int len = 0;
//      getHardwareInfo(hardwareInfo, &len);
//      encrypt(key, iv, hardwareInfo, len, hipher);
//      writeFile(filePath, hipher, len);
//  }
//
//  strncpy((unsigned char *) key, "RFbD56TI2smTyVsGd5Xav0yu99ZAMPTA", 32);
//  strncpy((unsigned char *) iv, "RFbD56TI2smTyVsG", 16);
//  int ok = CheckHardware(key, iv, "sys.dat");
//  printf("Is ok: %d \n", ok);

//  pthread_create(&tid_, NULL, _serve_thread_t<DvService, &DvService::Serve>, (void*) this);
//  pthread_t tid;
//  pthread_create(&tid, NULL, check, NULL);
//  pthread_join(tid, NULL);
}
