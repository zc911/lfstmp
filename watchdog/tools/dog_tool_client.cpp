/*
 * dog_tool_client.cpp
 *
 *  Created on: Apr 8, 2016
 *      Author: jiajiachen
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<sys/ptrace.h>

#include <glog/logging.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/rsa.h>
#include <openssl/evp.h>
#include <openssl/x509.h>
#include <openssl/pem.h>
#include "../src/common.h"
#include "../src/dog.h"
#include "../src/utils/tool.h"
#include "../src/include/watch_dog.h"
#define MAX_LEN 30000
#define BATCH_LEN 100
char * mergeHardwareFromStr(char *hardware_info, unsigned char *src, int len) {

     memcpy(hardware_info, &len, sizeof(len));
     hardware_info += sizeof(len);
     memcpy(hardware_info, src, len);
     hardware_info += len;

     return hardware_info;
}
char * mergeHardwareFromInf(char *hardware_info, const int &src) {
     memcpy(hardware_info, &src, sizeof(src));
     hardware_info += sizeof(src);
     return hardware_info;

}
int isTraced() {
     if (ptrace(PTRACE_TRACEME,0,0,0) < 0) {
          return -1;
     } else {
          return 1;
     }

}
int get_serialize_hardwareinfo(char *hardware_info) {
     int cpuIdLen, cpuInfoLen, cpuNum=0, biosUUIDLen, macLen, gpuIdLen,
               gpuInfoLen, totalLen, gpuNum=0;

     HardwareInfo info;
  //   memset(&info, 0, sizeof(info));

     char *begin = hardware_info;

     getCpuInfo(info.cpuInfo, cpuInfoLen, cpuNum);
     getBiosUUID(info.biosUUID, biosUUIDLen);
     getMac(info.mac, macLen);
     getGpuInfo(info.gpuInfo, gpuInfoLen, gpuNum);

     if(isTraced()<1)
          exit(-1);
     hardware_info = mergeHardwareFromStr(hardware_info, info.biosUUID,
                                          biosUUIDLen);
     hardware_info = mergeHardwareFromStr(hardware_info, info.mac, macLen);
     hardware_info = mergeHardwareFromInf(hardware_info, cpuNum);

     hardware_info = mergeHardwareFromStr(hardware_info, info.cpuInfo, cpuInfoLen);

     hardware_info = mergeHardwareFromInf(hardware_info, gpuNum);
     hardware_info = mergeHardwareFromStr(hardware_info, info.gpuInfo, gpuInfoLen);

     int len = hardware_info - begin;
     return len;
}
int rsa_encrypted(char *dst, char *src, int len) {

     int ret = 1;

     char key[] =
               "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtxUtu3+zqWNgA07rmah6\nwGl13Z4QfcuRWVx9UzEkYSnCT1arO8whMel2DzjGgUXC6DMiQlp8t9t43yfX5AW6\ntkOLHHjYAtDL+L9rh4dDYCBsdiNanA32Nvw+3FKcp8yYFBWbJDuqWb+OtcxIcW5N\nyJXEfgWWICVAIvd7vBf8zqAosrBDSTR7Q87nP2HlNvFT0gFguSxTzwFguaERSYVp\ng481xBhGztzlacXdMelINuxB6ELCkgliACUyrZwWpd6SE1liB3mo75C/0YPzvDpz\n+bz9GQte3DG3HFYi7TsogplwJVpUEmtzmJCkRyM18mxkIn63aOgwDnSBcM1f9tTS\nNwIDAQAB\n-----END PUBLIC KEY-----\n";
     BIO *keybio = BIO_new_mem_buf(key, -1);
     if (keybio == NULL) {
          printf("Failed to create key bio");
          return 0;
     }
     RSA *rsa = RSA_new();
     rsa = PEM_read_bio_RSA_PUBKEY(keybio, &rsa, NULL, NULL);
     ret = RSA_public_encrypt(len, (const unsigned char *) src,
                              (unsigned char *) dst, rsa,
                              RSA_PKCS1_PADDING);
     if (ret == -1) {
          printf("encrypt error!... \n");
          return -1;

     }
     return ret;

}
void writeFile(const char *fn, char *en, int len) {
     FILE *fp = fopen(fn, "wb");
     if (fp == NULL) {
          printf("Unable\n");
     }
     fwrite(en, len, sizeof(char), fp);
     fclose(fp);
}

int getSerializeEncryptedData(char *dst, char *src, int total_len) {
     int after_total_len = 0;
     for (int i = 0; i < total_len; i += BATCH_LEN) {
          int ret;
          int len = (i + BATCH_LEN) > total_len ? (total_len - i) : BATCH_LEN;
          dst += sizeof(len);

          ret = rsa_encrypted(dst, src, len);
          if (ret == -1) {
               return -1;
          }
          memcpy(dst - sizeof(ret), &ret, sizeof(ret));

          src += len;
          dst += ret;

          after_total_len += sizeof(ret);
          after_total_len += ret;
     }

     return after_total_len;
}
int main(int argc, char *argv[]) {

     char *src = (char *) calloc(MAX_LEN, sizeof(char));
     char *dst = (char *) calloc(MAX_LEN, sizeof(char));
     int len = get_serialize_hardwareinfo(src);

     int ret = getSerializeEncryptedData(dst, src, len);
     if (ret < 0) {
          return -1;
     }
     writeFile("deepv.dat", dst, ret);
     delete src;
     delete dst;
     printf("license 'deepv.dat' has been saved in the current director\n");
     exit(-1);
}
