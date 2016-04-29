#include <stdio.h>
#include <stdlib.h>
#include <glog/logging.h>
#include "../src/common.h"
#include "../src/dog.h"
#include "../src/utils/tool.h"
#include "../src/include/watch_dog.h"

extern const unsigned int AES_KEY_LEN;

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
void readReservedMsg(){
     DogData data;
     uint32_t code,time;
     KeyStatus s = readReservedFromDog(0,code,time);
     if (s != KEY_OK) {
         printf("Read reserved data failed\n");
         exit(-1);
     }
     printf("error code: %d \ntime: %d",code,time);

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
    for (int i = 1; i <= FEATURE_CAR_MARKER; i = i * 2) {
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
                printf("Invalid input, please retry \n");
                break;
            }
        }
        printMenu();
    }

    return 0;
}
