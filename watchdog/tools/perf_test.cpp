#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include "../src/include/watch_dog.h"
#include "../src/dog.h"

int threadNum = 100;
int interval = 1;

void* checkDog(void *p) {
    while (1) {
//        printf("Check hardware...\n");
//        if (CheckHardware() != ERR_SUCCESS) {
//            printf("Check hardware failed \n");
//        }
        loginDog(0);
        logoutDog(0);
        sleep(interval);
    }
    return NULL;
}

int main(int argc, char *argv[]) {

    threadNum = atoi(argv[1]);
    interval = atoi(argv[2]);

    for (int i = 0; i < threadNum; ++i) {
        pthread_t id;
        pthread_create(&id, NULL, checkDog, NULL);
//        sleep(1);

    }
    while(1){
        sleep(100000);
    }
    return 0;

}
