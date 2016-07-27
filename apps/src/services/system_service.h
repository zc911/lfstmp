/*============================================================================
 * File Name   : system_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_SYSTEM_SERVICE_H_
#define MATRIX_APPS_SYSTEM_SERVICE_H_
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <thread>

#include <glog/logging.h>
#include <boost/algorithm/string.hpp>
#include "config/config_val.h"
#include "config.h"
#include "model/system.grpc.pb.h"
#include "string_util.h"
namespace dg {
static int rx = 0;
static int tx = 0;
using namespace ::dg::model;
//static void networkInfo(int *rx, int *tx) {
//    char id[1000];
//    while (1) {
//        struct timeval n;
//        gettimeofday(&n, NULL);
//        uint64_t start = n.tv_sec * 1000 + n.tv_usec / 1000;
//        uint64_t start_rx = 0;
//        uint64_t start_tx = 0;
//
//        memset(id, 0, sizeof(id));
//        FILE *out = popen("ifconfig | grep bytes: |egrep -o ':[0-9]+'", "r");
//        if (out == NULL) {
//            return;
//        }
//
//        fgets(id, sizeof(id), out);
//        if (id[0] == ':')
//            start_rx = atoi(id + 1);
//        memset(id, 0, sizeof(id));
//        fgets(id, sizeof(id), out);
//        if (id[0] == ':')
//            start_tx = atoi(id + 1);
//        fclose(out);
//        sleep(1);
//
//        uint64_t end_rx = 0;
//        uint64_t end_tx = 0;
//
//        memset(id, 0, sizeof(id));
//        out = popen("ifconfig | grep bytes: |egrep -o ':[0-9]+'", "r");
//        if (out == NULL) {
//            fclose(out);
//            return;
//        }
//
//        fgets(id, sizeof(id), out);
//        if (id[0] == ':')
//            end_rx += atoi(id + 1);
//        memset(id, 0, sizeof(id));
//        fgets(id, sizeof(id), out);
//        fclose(out);
//        if (id[0] == ':')
//            end_tx += atoi(id + 1);
//        gettimeofday(&n, NULL);
//        uint64_t end = n.tv_sec * 1000 + n.tv_usec / 1000;
//
//        *rx = (end_rx - start_rx) / (end - start);
//        *tx = (end_tx - start_tx) / (end - start);
//
//    }
//}

class SystemAppsService {

public:
    SystemAppsService(const Config *config, string name, int baseId = 0);
    virtual ~SystemAppsService();

    MatrixError Ping(const PingRequest *request, PingResponse *response);

    MatrixError SystemStatus(const SystemStatusRequest *request,
                             SystemStatusResponse *response);

    MatrixError GetInstances(const GetInstancesRequest *request,
                             InstanceConfigureResponse *response);

    MatrixError ConfigEngine(const InstanceConfigureRequest *request,
                             InstanceConfigureResponse *response);

    static int getCpuUsage(std::string &msg) {
        char id[300];
        FILE *out = popen("top -bn1 |grep 'Cpu(s)'", "r");
        if (out == NULL) {
            fclose(out);
            return -1;
        }
        fgets(id, sizeof(id), out);
        pclose(out);

        std::vector<std::string> strs;
        splitSpace(strs, string(id));
        if (strs.size() >= 2) {
            msg = strs[1] + "%";
        } else {
            return -1;
        }
        return 1;
    }
    static int getMemInfo(std::string &msg, std::string cmd) {
        std::string fullcmd = "cat /proc/meminfo |grep " + cmd;
        char id[50];
        FILE *out = popen(fullcmd.c_str(), "r");
        if (out == NULL) {
            fclose(out);
            return -1;
        }

        fgets(id, sizeof(id), out);
        pclose(out);

        std::vector<std::string> strs;
        splitSpace(strs, string(id));
        if (strs.size() >= 2) {
            msg = strs[1] + "kB";
        } else {
            return -1;
        }
        return 1;
    }
    static int getDiskInfo(std::string &msg, std::string cmd) {
        char id[1000];
        FILE *out = popen("df / ", "r");
        if (out == NULL) {
            fclose(out);
            return -1;
        }

        fgets(id, sizeof(id), out);
        std::vector<std::string> keys;
        splitSpace(keys, string(id));
        int index;
        for (index = 0; index < keys.size(); index++) {
            if (keys[index] == cmd)
                break;
        }

        memset(id, 0, sizeof(id));

        fgets(id, sizeof(id), out);
        pclose(out);
        std::vector<std::string> values;
        splitSpace(values, string(id));

        if (values.size() >= index) {
            msg = values[index] + "kB";
        } else {
            return -1;
        }
        return 1;

    }
    static int getGPUMemory(std::string &msg, std::string cmd) {
        char id[1000];
        FILE *out = popen("nvidia-smi -L", "r");
        if (out == NULL) {
            fclose(out);
            return -1;
        }
        int gpuCnt = 0;
        while (fgets(id, sizeof(id), out) != NULL) {
            gpuCnt++;
        }
        memset(id, 0, sizeof(id));
        out = popen("nvidia-smi |grep MiB", "r");
        if (out == NULL) {
            fclose(out);
            return -1;
        }
        vector<vector<string> > memgpus;
        for (int i = 0; i < gpuCnt; i++) {
            fgets(id, sizeof(id), out);
            vector<string> memgpu;
            splitSpace(memgpu, id);
            memgpus.push_back(memgpu);
            if (cmd == "Used") {
                msg = "Deviced " + to_string(i) + ": " + memgpu[8];
            } else if (cmd == "Total") {
                msg = "Deviced " + to_string(i) + ": " + memgpu[10];
            }
            if (i != gpuCnt - 1)
                msg += "\n";
        }
        return 1;

    }

    int getNetworkInfo(std::string &msg, std::string cmd) {
        if (cmd == "RX") {
            msg = to_string(rx) + " kB/s";
        } else if (cmd == "TX") {
            msg = to_string(tx) + " kB/s";

        }
        return 1;
    }
private:
    void initNetworkThread();
    const Config *config_;
    string name_;
    string modelversion_;
    string serviceversion_;

};
};

#endif //MATRIX_APPS_SYSTEM_SERVICE_H_
