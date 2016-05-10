/*
 * verification_tool.cpp
 *
 *  Created on: Apr 13, 2016
 *      Author: jiajiachen
 */
#include <stdio.h>
#include <stdlib.h>
#include <glog/logging.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include "../src/common.h"
#include "../src/dog.h"
#include "../src/utils/tool.h"
#include "../src/include/watch_dog.h"
#define MAX_LEN 30000
void writeHardwareInfo(HardwareInfo &info) {
     printf("Write hardware info ...\n");
     KeyStatus s = writeHwInfoIntoDog(0, &info);
     if (s != KEY_OK) {
          printf("Write hardware info failed\n");
          exit(-1);
     }
}
unsigned char * splitHardwareFromStr(unsigned char *dst, unsigned char *src,
                                     int len) {
     memcpy(&len, src, sizeof(len));
     src += sizeof(len);
     memcpy(dst, src, len);
     dst[len] = 0;
     src += (len);
     return src;
}
unsigned char * splitHardwareFromInt(int &dst, unsigned char *src) {
     memcpy(&dst, src, sizeof(dst));
     src += sizeof(dst);
     return src;
}
void printMessage(HardwareInfo &hinfo, int &gpuNum, unsigned char *gpuInfo) {
     printf("\n++++++++HARDWARE INFO+++++++++++\n");
     printf("CPU%s\n", hinfo.cpuInfo);
     printf("--------------------------------\n");
     printf("BIOS%s\n", hinfo.biosUUID);
     printf("--------------------------------\n");

     printf("Mac     ");
     for (int i = 0; i < strlen((char *) hinfo.mac) - 1; i++) {
          printf("%X-", hinfo.mac[i]);
     }

     printf("%X\n", hinfo.mac[strlen((char *) hinfo.mac) - 1]);
     printf("--------------------------------\n");

     printf("GPUinfo     %s\n", gpuInfo);
     printf("--------------------------------\n");

     printf("GPUid     %s\n", hinfo.gpuInfo);
     printf("--------------------------------\n");

     printf("GPUNum  %d\n", gpuNum);

     printf("++++++++HARDWARE INFO+++++++++++\n");

     printf("Give license to this machine?(y/n)\n");

}
void write_hardwareinfo(unsigned char *hardware_info) {
     HardwareInfo hinfo;
     unsigned char gpuInfo[3000];
     memset(gpuInfo, 0, 3000);
     unsigned char cpuInfo[3000];
     memset(cpuInfo, 0, 3000);

     int cpuIdLen, biosUUIDLen, macLen, cpuInfoLen, gpuIdLen, gpuInfoLen,
               cpuNum, gpuNum;

     hardware_info = splitHardwareFromStr(hinfo.biosUUID, hardware_info,
                                          biosUUIDLen);
     hardware_info = splitHardwareFromStr(hinfo.mac, hardware_info, macLen);
     hardware_info = splitHardwareFromInt(cpuNum, hardware_info);
     hardware_info = splitHardwareFromStr(cpuInfo, hardware_info, cpuInfoLen);
     hardware_info = splitHardwareFromStr(hinfo.cpuInfo, hardware_info, cpuIdLen);
     hardware_info = splitHardwareFromInt(gpuNum, hardware_info);
     hardware_info = splitHardwareFromStr(gpuInfo, hardware_info, gpuInfoLen);
     hardware_info = splitHardwareFromStr(hinfo.gpuInfo, hardware_info, gpuIdLen);

     printMessage(hinfo, gpuNum, gpuInfo);

     int input = 0;

     input = getchar();
     switch (input) {
          case 'y':
          case 'Y':
               writeHardwareInfo(hinfo);
               break;
     }

}
int rsa_decrypted(char *dst, const char *src, int len) {
     printf("Read encryped file ...\n");

     FILE *fp = fopen("private_unencrypted.pem", "rb");
     if (fp == 0) {
          printf("Open key file error ...\n");
          return -1;
     }
     RSA *rsa = RSA_new();
     rsa = PEM_read_RSAPrivateKey(fp, &rsa, NULL, NULL);
     fclose(fp);

     if (rsa == 0) {
          printf("Read key error ...\n");
          return -1;
     }

     int rsa_len;
     rsa_len = RSA_size(rsa);
     int ret = RSA_private_decrypt(len, (unsigned char *) src,
                                   (unsigned char *) dst, rsa,
                                   RSA_PKCS1_PADDING);
     if (ret < 0) {
          printf("Encrypt error ...\n");
          return -1;
     }
     return ret;
     // write_hardwareinfo(p_de);
}
int readFile(const char *fn, char *dst) {

     FILE *hardware_file = fopen(fn, "rb");
     if (hardware_file == 0) {
          printf("Open hardware file error\n");
          return -1;
     }
     int ret;
     ret = fread(dst, sizeof(char), MAX_LEN, hardware_file);

     fclose(hardware_file);
     return ret;
}
int getDeserializeEncryptedData(char *dst, char *src, int total_len) {
     char *begin = dst;
     int size;
     for (int i = 0; i < total_len; i += sizeof(size)) {
          memcpy(&size, src, sizeof(size));
          src += sizeof(size);
          int ret = rsa_decrypted(dst, src, size);

          dst += ret;
          src += size;
          i += size;
     }
     return dst - begin;
}
int main(int argc, char *argv[]) {
     char *src = (char *) calloc(MAX_LEN, sizeof(unsigned char));
     char *dst = (char *) calloc(MAX_LEN, sizeof(unsigned char));

     int len = readFile("deepv.dat", (char *) src);
     int ret = getDeserializeEncryptedData(dst, src, len);
     write_hardwareinfo((unsigned char *) dst);

     return 1;
}

