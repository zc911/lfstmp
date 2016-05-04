#include <stdio.h>
#include <stdlib.h>
#include <glog/logging.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include "../src/common.h"
#include "../src/dog.h"
#include "../src/utils/tool.h"
#include "../src/include/watch_dog.h"

extern const unsigned int AES_KEY_LEN;
#define MAX_LEN 30000

void writeHardwareInfo() {
     printf("Read hardware info ...\n");
     HardwareInfo info;
     getHardwareInfo(info);

     printf("Write hardware info ...\n");
     KeyStatus s = writeHwInfoIntoDog(0, &info);
     if (s != KEY_OK) {
          printf("Write hardware info failed\n");
          exit(-1);
     }
}
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
     dst[len] = '\0';
     src += (len);
     return src;
}
unsigned char * splitHardwareFromInt(int &dst, unsigned char *src) {
     memcpy(&dst, src, sizeof(dst));
     src += sizeof(dst);
     return src;
}
void printMessage(HardwareInfo &hinfo, int &gpuNum) {
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

     printf("GPUinfo %s\n", hinfo.gpuInfo);
     printf("--------------------------------\n");

     printf("--------------------------------\n");

     printf("GPUNum  %d\n", gpuNum);

     printf("++++++++HARDWARE INFO+++++++++++\n");

     printf("Give license to this machine?(y/n)\n");

}
int write_hardwareinfo(unsigned char *hardware_info) {
     HardwareInfo hinfo;

     int cpuIdLen, biosUUIDLen, macLen, cpuInfoLen, gpuIdLen, gpuInfoLen,
               cpuNum, gpuNum;

     hardware_info = splitHardwareFromStr(hinfo.biosUUID, hardware_info,
                                          biosUUIDLen);
     hardware_info = splitHardwareFromStr(hinfo.mac, hardware_info, macLen);
     hardware_info = splitHardwareFromInt(cpuNum, hardware_info);
     hardware_info = splitHardwareFromStr(hinfo.cpuInfo, hardware_info, cpuInfoLen);
     hardware_info = splitHardwareFromInt(gpuNum, hardware_info);
     hardware_info = splitHardwareFromStr(hinfo.gpuInfo, hardware_info, gpuInfoLen);

     printMessage(hinfo, gpuNum);

     int input = 0;

     input = getchar();
     switch (input) {
          case 'y':
          case 'Y':
               writeHardwareInfo(hinfo);
               break;
          default:
               return 0;

     }
     return 1;

}
int rsa_decrypted(char *dst, const char *src, int len) {
   //  printf("Read encryped file ...\n");

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
          if(ret<0)
               return -1;
          dst += ret;
          src += size;
          i += size;
     }
     return dst - begin;
}
int licenseHardware() {
     char *src = (char *) calloc(MAX_LEN, sizeof(unsigned char));
     char *dst = (char *) calloc(MAX_LEN, sizeof(unsigned char));

     int len = readFile("deepv.dat", (char *) src);
     int ret = getDeserializeEncryptedData(dst, src, len);
     if (ret < 0) {
          printf("decrypt data error ...\n");
          return -1;
     }
     if(!write_hardwareinfo((unsigned char *) dst)){
          return 0;
     }
     return 1;
}
void saveOldModelKey() {

     FILE *f = fopen("model_key.perm", "rb");
     if (f == NULL) {
          printf("There is no modek_key.perm exits. \n");
          return;
     }
     time_t now = time(NULL);
     struct tm *tmNow = gmtime(&now);
     char bakName[128];
     sprintf(bakName, "model_key.perm.%d_%d_%d_%d:%d:%d.bak\0", tmNow->tm_year,
             tmNow->tm_mon + 1, tmNow->tm_mday, tmNow->tm_hour, tmNow->tm_min,
             tmNow->tm_sec);
     FILE *bakFile = fopen(bakName, "wb");
     if (bakFile == NULL) {
          printf("Create model_key.perm.bak file failed \n");
          exit(-1);
     }
     unsigned char key[AES_KEY_LEN];
     int n = fread(key, sizeof(unsigned char), sizeof(key), f);
     if (n != AES_KEY_LEN) {
          printf("Invalid key len: %d, should be: %d \n", n, AES_KEY_LEN);
          exit(-1);
     }
     printf("Save the old key to %s \n", bakName);
     fwrite(key, sizeof(unsigned char), sizeof(key), bakFile);
     fflush(bakFile);
     fclose(bakFile);
     fclose(f);

}

void genNewModelKey() {
     unsigned char key[AES_KEY_LEN];
     generateKey(key, AES_KEY_LEN);
     FILE *f = fopen("model_key.perm", "wb");
     if (f == NULL) {
          printf("Create model key file error \n");
          exit(-1);
     }
     fwrite(key, sizeof(unsigned char), sizeof(key), f);
     fflush(f);
     fclose(f);
     printf("Write the new model key to model_key.perm\n");
}

void writeModelKey() {
     FILE *f = fopen("model_key.perm", "rb");
     if (f == NULL) {
          printf("Open model key file failed \n");
          exit(-1);
     }
     unsigned char key[AES_KEY_LEN];
     int n = fread(key, sizeof(unsigned char), sizeof(key), f);
     if (n != AES_KEY_LEN) {
          printf("Invalid key len: %d, should be: %d \n", n, AES_KEY_LEN);
          exit(-1);
     }
     KeyStatus s = writeModelKeyIntoDog(0, key, sizeof(key));
     if (s != KEY_OK) {
          printf("Write model key to dog failed\n");
          exit(-1);
     }

}
void readReservedMsg() {
     DogData data;
     uint32_t code, time;
     KeyStatus s = readReservedFromDog(0, code, time);
     if (s != KEY_OK) {
          printf("Read reserved data failed\n");
          exit(-1);
     }
     printf("error code: %d \ntime: %d", code, time);

}
void readDog() {
     DogData data;
     KeyStatus s = readDogData(0, data);
     if (s != KEY_OK) {
          printf("Read dog failed \n");
          exit(-1);
     }
     printHex(&data, sizeof(data), "Dog data");
}

void writeFeatures() {
     printf("\nPlease input feature values: ");
     int value;
     if (scanf("%d", &value) == EOF)
          return;
     if (value < 0)
          value = 0;
     getchar();
     if (writeFeaturesIntoDog(0, value) == KEY_OK) {
          printf("Write features into dog successfully \n");
     } else {
          printf("Write features into dog failed \n");
     }

}

void showFeatures() {
     FeatureSet fs;
     if (GetFeatureSet(fs) == ERR_FEATURE_ERROR) {
          printf("Get features from dog failed. \n");
          return;
     }
     printf("====Feature Sets====\n");
     for (int i = 1; i <= FEATURE_FACE_RANKER; i = i * 2) {
          string name;
          switch (i) {
               case 1:
                    name = "FEATURE_CAR_DETECTION";
                    break;
               case 2:
                    name = "FEATURE_RESERVED";
                    break;
               case 4:
                    name = "FEATURE_CAR_STYLE";
                    break;
               case 8:
                    name = "FEATURE_CAR_COLOR";
                    break;
               case 16:
                    name = "FEATURE_CAR_PLATE";
                    break;
               case 32:
                    name = "FEATURE_CAR_MARKER";
                    break;
               case 64:
                    name = "FEATURE_CAR_RANKER";
                    break;
               case 128:
                    name = "FEATURE_CAR_EXTRACTOR";
                    break;
               case 256:
                    name = "FEATURE_FACE_EXTRACTOR";
                    break;
               case 512:
                    name = "FEATURE_FACE_RANKER";
                    break;
               default:
                    break;

          }
          printf("    %s: ", name.c_str());
          if (CheckFeature((Feature) i, fs) == ERR_FEATURE_ON) {
               printf(" ON \n");
          } else {
               printf(" OFF \n");
          }
     }

}

void printMenu() {
     printf("\n----- WatchDog Tool Menu -----\n");
     printf("1: Read dog content \n");
     printf("2: Show features \n");
     printf("3: Write hardware info \n");
     printf("4: Write features \n");
     printf("5: Write model key \n");
     printf("6: Read reserved data \n");
     printf("l: License this hardware \n");
     printf("g: Generate a new model key \n");
     printf("r: Reset dog \n");
     printf("q: Quit\n");
     printf("Please input [1|2|3|4|5|g|r|q]: ");
}

bool reAsk(const char *msg) {
     printf("This operation will %s, are you sure [y/n]: ", msg);
     int input = getchar();
     getchar();
     if (input == 'y') {
          return true;
     }
     return false;
}

int main(int argc, char *argv[]) {
     google::InitGoogleLogging(argv[0]);
     int input = 0;
     printMenu();

     while ((input = getchar()) != 'q') {
          getchar();
          switch (input) {
               case '1': {
                    readDog();
                    break;
               }
               case '2': {
                    showFeatures();
                    break;
               }
               case '3': {
                    writeHardwareInfo();
                    break;
               }
               case '4': {
                    writeFeatures();
                    break;
               }
               case '5': {
                    writeModelKey();
                    break;
               }
               case '6': {
                    readReservedMsg();
                    break;
               }
               case 'l': {
                    if(licenseHardware()){
                         printf("Successful ...\n");
                    }
                    return 0;
               }
               case 'g': {
                    if (!reAsk("regenerate model key"))
                         break;
                    saveOldModelKey();
                    genNewModelKey();
                    break;
               }
               case 'r': {
                    resetDogData(0);
                    break;
               }
               default: {
                    break;
               }
          }
          printMenu();
     }

     return 0;
}
