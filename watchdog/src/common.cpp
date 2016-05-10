/*
 * dog.c
 *
 *  Created on: Nov 29, 2015
 *      Author: chenzhen
 */

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <openssl/aes.h>
#include <openssl/md5.h>
#include <glog/logging.h>

#include "common.h"
#include "utils/tool.h"

using namespace std;

void getCpuId(unsigned char *cpuId, int &len) {

     char id[256];
     len = 0;
     FILE *out = popen("dmidecode -t 4 | grep ID", "r");
     if (out == NULL) {
          fclose(out);

          return;
     }

     while (fgets(id, sizeof(id), out) != NULL) {
          int currentLen = strlen(id) - 1;
          memcpy(cpuId + len, (unsigned char*) id, currentLen);
          printf("id %s\n",id);
          len += currentLen;
     }



     pclose(out);
}
void getCpuInfo(unsigned char *cpuVersion, int &len, int &cpuNum) {
     cpuNum=0;
     char version[256];
     memset(version,0,256);
     len = 0;
     FILE *out = popen("dmidecode -t 4 | grep -E 'Version|ID'", "r");
     if (out == NULL) {
          fclose(out);

          return;
     }

     while (fgets(version, sizeof(version), out) != NULL) {
          int currentLen = strlen(version) - 1;
          memcpy(cpuVersion + len, (unsigned char*) version, currentLen);
          printf("cpuVersion %s\n",cpuVersion);

          len += currentLen;
          cpuNum++;
     }
     cpuNum/=2;
     pclose(out);
}

void getBiosUUID(unsigned char *uuid, int &len) {

     char id[256];
     len = 0;

         FILE *out = popen("dmidecode -t 1 | grep UUID", "r");
     if (out == NULL) {
          fclose(out);

          return;
     }
    fgets(id, sizeof(id), out);
     len = strlen(id) - 1;
     if (len > sizeof(id) || len < 0) {
          len = sizeof(id);
     }
     memcpy(uuid, (unsigned char*) id, len);


     pclose(out);
}
void getGpuInfo(unsigned char *gpuinfo, int &len, int &gpuNum) {

     char info[256];
     len = 0;

     FILE *out = popen("nvidia-smi -L", "r");
     if (out == NULL) {
          fclose(out);

          return;
     }
     while (fgets(info, sizeof(info), out) != NULL) {
          int currentLen = strlen(info) - 1;
          memcpy(gpuinfo + len, (unsigned char*) info, currentLen);
          len += currentLen;
          gpuNum++;
     }
     pclose(out);
}

void getGpuId(unsigned char *gpuId, int &len, int &gpuNum) {
     char id[256];
     len = 0;
     gpuNum = 0;
     FILE *out = popen("nvidia-smi -L | awk '{print $NF}'", "r");
     if (out == NULL) {
          fclose(out);

          return;
     }
     while (fgets(id, sizeof(id), out) != NULL) {
          if (strlen(id) > 16) {
               char prefix[4];
               strncpy(prefix, id, 3);
               prefix[3] = '\0';
               if (strcmp(prefix, "GPU") == 0) {
                    int currentLen = strlen(id) - 2;
                    if (len + currentLen > 1024) {
                         break;
                    }
                    memcpy(gpuId + len, (unsigned char*) id, currentLen);
                    len += currentLen;
                    gpuNum++;
               }
          }
     }
     pclose(out);

}

void getMac(unsigned char *mac, int &len) {

     struct ifreq req;
     int sock;
     len = 0;

     if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
          perror("create socket error");

          return;
     }

     strcpy(req.ifr_name, "bond0");
     if (ioctl(sock, SIOCGIFHWADDR, &req) < 0) {
          strcpy(req.ifr_name, "eth0");
          if (ioctl(sock, SIOCGIFHWADDR, &req) < 0) {
               strcpy(req.ifr_name, "em1");
               if (ioctl(sock, SIOCGIFHWADDR, &req) < 0) {
                    perror("ioctl error");
                    return;
               }

          }
     }

     int i;
     for (i = 0; i < 6; ++i) {
          unsigned char s = (unsigned char) req.ifr_hwaddr.sa_data[i];
          memcpy(mac++, &s, 1);
          (len)++;
     }

}

void getHardwareInfo(HardwareInfo &hardware) {

     int cpuIdLen, biosUUIDLen, macLen, gpuIdLen, totalLen;
     int gpuNum;

     memset(&hardware, 0, sizeof(hardware));

     getCpuId(hardware.cpuInfo, cpuIdLen);
     getBiosUUID(hardware.biosUUID, biosUUIDLen);
     getMac(hardware.mac, macLen);
     getGpuId(hardware.gpuInfo, gpuIdLen, gpuNum);

     hardware.cpuInfo[cpuIdLen] = '\0';
     hardware.biosUUID[biosUUIDLen] = '\0';
     hardware.mac[macLen] = '\0';
 //    hardware.gpuId[gpuIdLen] = gpuNum;
     hardware.gpuInfo[gpuIdLen] = '\0';

}

int fingerprint(HardwareInfo *info, unsigned char *figerprint) {
     unsigned char md5[FINGER_PRINT_LEN] = { 0 };
     MD5_CTX ctx;
     MD5_Init(&ctx);

     MD5_Update(&ctx, info, sizeof(*info));
     MD5_Final(md5, &ctx);

     memcpy(figerprint, md5, FINGER_PRINT_LEN);
     return 0;
}

void writeFile(const char* filePath, unsigned char *content, const int len) {
     FILE *file = fopen(filePath, "wb");
     if (file == NULL) {
          printf("Open file error");
          return;
     }
     fwrite(content, 1, len, file);
     fclose(file);
}

void readFile(const char* filePath, unsigned char *content, int *len) {
     FILE *file = fopen(filePath, "rb");
     if (file == NULL) {
          return;
     }
     *len = fread(content, 1, *len, file);
     fclose(file);
}

int compare(const unsigned char *a, const int lenA, const unsigned char *b,
            const int lenB) {
     if (lenA != lenB) {
          return -1;
     }

     return memcmp(a, b, lenA);
}

int aesEncrypt(unsigned char *modelData, const unsigned long len,
               unsigned char *enModelData, unsigned char *modelKey) {
     unsigned char iv[AES_IV_LEN];
     memcpy(iv, modelKey + 1, AES_IV_LEN);

     AES_KEY aes_enc;
     if (AES_set_encrypt_key(modelKey, 128, &aes_enc) < 0) {
          LOG(ERROR)<<"KEY ISN'T CORRECT"<<endl;
          return -1;
     }

     AES_cbc_encrypt(modelData, enModelData, len, &aes_enc, iv, AES_ENCRYPT);
     memset(modelKey, 0xff, AES_KEY_LEN);
     memset(iv, 0xff, AES_IV_LEN);
     return 0;
}

int aesDecrypt(unsigned char *enModelData, const unsigned long enLen,
               unsigned char *modelData, unsigned char *modelKey) {
     unsigned char iv[AES_IV_LEN];
     memcpy(iv, modelKey + 1, AES_IV_LEN);

     AES_KEY aes_enc;
     if (AES_set_decrypt_key(modelKey, 128, &aes_enc) < 0) {
          LOG(ERROR)<<"KEY ISN'T CORRECT"<<endl;
          return -1;
     }

     AES_cbc_encrypt(enModelData, modelData, enLen, &aes_enc, iv, AES_DECRYPT);
     memset(modelKey, 0xff, AES_KEY_LEN);
     memset(iv, 0xff, AES_IV_LEN);
     return 0;
}

