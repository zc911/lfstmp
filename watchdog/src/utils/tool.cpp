#include "tool.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

void printHex(void *data, const int len, const char *title) {
    printf("\n-------%s-------\n", title);
    for (int i = 0; i < len; ++i) {
        printf("%02x-", ((unsigned char *) data)[i]);
    }
    printf("\n-------%s-------\n", title);
}

void generateKey(unsigned char *key, int len) {

    srand((unsigned) time(NULL) * 100);
    for (int i = 0; i < len; ++i) {
        int r = rand();
        unsigned char c = (unsigned char) (r % 0xFF);
        memcpy(key + i, &c, 1);
    }

}
