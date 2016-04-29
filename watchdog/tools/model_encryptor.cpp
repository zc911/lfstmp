#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include "../src/include/watch_dog.h"

using namespace std;

void usage(const char *name) {
    printf("Usage: \n");
    printf("%s -i MODEL_FILE -o ENCRYPT_MODEL_FILE \n", name);
    printf("-i: input model file \n");
    printf("-o: output encrypted model file \n");
    printf("-a: AWS mode \n");
    printf("-h: help \n");
}

int main(int argc, char **argv) {

    int i = 0;
    string input, output;
    bool awsMode = false;
    while ((i = getopt(argc, argv, "i:o:ah")) != -1) {
        switch (i) {
            case 'i':
                input = string(optarg);
                break;
            case 'o':
                output = string(optarg);
                break;
            case 'a':
                awsMode = true;
                break;
            case 'h':
                usage(argv[0]);
                exit(-1);
                break;
            default:
                break;
        }
    }

    if (input.size() <= 0 || output.size() <= 0) {
        usage(argv[0]);
        exit(-1);
    }

    FILE *modelFile = fopen(input.c_str(), "rb");
    if (modelFile == NULL) {
        printf("Open model patch file failed \n");
        exit(-1);
    }

    int modelLen = 0;
    while (getc(modelFile) != EOF) {
        modelLen++;
    }
    rewind(modelFile);
    printf("The model patch len: %d \n", modelLen);
    if (modelLen % 16 != 0) {
        modelLen = modelLen + 16 - (modelLen % 16);
    }
    printf("The model patch len: %d \n", modelLen);
    unsigned char *modelData = (unsigned char *) malloc(modelLen);
    memset(modelData, 0, modelLen);
    unsigned char *enModelData = (unsigned char *) malloc(modelLen);
    memset(enModelData, 0, modelLen);

    if (modelData == NULL || enModelData == NULL) {
        printf("Malloc mem failed \n");
        exit(-1);
    }
    fread(modelData, sizeof(char), modelLen, modelFile);

    printf("Model data: \n");
    for (int i = modelLen - 100; i < modelLen; ++++i) {
        printf("%02x%02x ", modelData[i], modelData[i + 1]);
    }
    printf("\n");

    fclose(modelFile);
    int ret;
    if (awsMode) {
        ret = EncryptModelAWS(modelData, modelLen, enModelData);
    } else {
        ret = EncryptModel(modelData, modelLen, enModelData);
    }
    if (ret != ERR_SUCCESS) {
        free(modelData);
        free(enModelData);
        printf("Encrypt the model failed \n");
        exit(-1);
    }
    FILE *enModelFile = fopen(output.c_str(), "wb");
    if (enModelFile == NULL) {
        printf("Open encrypt model file failed \n");
        free(modelData);
        free(enModelData);
        exit(-1);
    }

    printf("En Model data: \n");
    for (int i = modelLen - 100; i < modelLen; ++++i) {
        printf("%02x%02x ", enModelData[i], enModelData[i + 1]);
    }
    printf("\n");

    fwrite(enModelData, sizeof(char), modelLen, enModelFile);
    fflush(enModelFile);
    fclose(enModelFile);
    printf("Write encrypt model file to %s successful. \n", output.c_str());

}
