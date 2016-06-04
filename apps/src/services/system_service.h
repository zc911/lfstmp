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

#include <glog/logging.h>
#include <boost/algorithm/string.hpp>
#include "config/config_val.h"
#include "config.h"
#include "string_util.h"
#include "model/system.grpc.pb.h"

namespace dg {
using namespace ::dg::model;

class SystemAppsService {

 public:
    SystemAppsService(const Config *config);
    virtual ~SystemAppsService();

    MatrixError Ping(const PingRequest *request, PingResponse *response);

    MatrixError SystemStatus(const SystemStatusRequest *request,
                             SystemStatusResponse *response);

    MatrixError GetInstances(const GetInstancesRequest *request,
                             InstanceConfigureResponse *response);

    MatrixError ConfigEngine(const InstanceConfigureRequest *request,
                             InstanceConfigureResponse *response);
    static int getCpuUsage(std::string &msg){
        char id[300];
        FILE *out = popen("top -bn1 |grep 'Cpu(s)'", "r");
        if (out == NULL) {
             fclose(out);
             return -1;
        }
        fgets(id, sizeof(id), out);
        pclose(out);

        std::vector<std::string> strs;
        splitSpace(strs,string(id));
        if(strs.size()>=2){
            msg=strs[1]+"%";
        }else{
            return -1;
        }
        return 1;
    }
    static int getMemInfo(std::string &msg,std::string cmd){
        std::string fullcmd = "cat /proc/meminfo |grep "+cmd;
        char id[50];
        FILE *out = popen(fullcmd.c_str(), "r");
        if (out == NULL) {
             fclose(out);
             return -1;
        }

        fgets(id, sizeof(id), out);
        pclose(out);

        std::vector<std::string> strs;
        splitSpace(strs,string(id));
        if(strs.size()>=2){
            msg=strs[1]+"kB";
        }else{
            return -1;
        }
        return 1;
    }
    static int getDiskInfo(std::string &msg,std::string cmd){
        char id[1000];
        FILE *out = popen("df / ", "r");
        if (out == NULL) {
             fclose(out);
             return -1;
        }


        fgets(id, sizeof(id), out);
        std::vector<std::string> keys;
        splitSpace(keys,string(id));
        int index;
        for(index=0;index<keys.size();index++){
            if(keys[index]==cmd)
                break;
        }

        memset(id,0,sizeof(id));

        fgets(id, sizeof(id), out);
        pclose(out);
        std::vector<std::string> values;
        splitSpace(values,string(id));

        if(values.size()>=index){
            msg=values[index];
        }else{
            return -1;
        }
        return 1;

    }
    static int getGPUMemory(std::string &msg,std::string cmd){
        char id[1000];
        FILE *out = popen("nvidia-smi -L", "r");
        if (out == NULL) {
             fclose(out);
             return -1;
        }
        int gpuCnt=0;
        while(fgets(id,sizeof(id),out)!=NULL){
            gpuCnt++;
        }
        memset(id,0,sizeof(id));
        out=popen("nvidia-smi |grep MiB","r");
        if(out==NULL){
            fclose(out);
            return -1;
        }
        vector<vector<string> > memgpus;
        for(int i=0;i<gpuCnt;i++){
            fgets(id,sizeof(id),out);
            vector<string> memgpu;
            splitSpace(memgpu,id);
            memgpus.push_back(memgpu);
            if(cmd=="Used"){
                msg="Deviced "+to_string(i)+": "+memgpu[8];
            }else if(cmd=="Total"){
                msg="Deviced "+to_string(i)+": "+memgpu[10];
            }
            if(i!=gpuCnt-1)
            msg+="\n";
        }

    }
 private:
    const Config *config_;

};
}
;

#endif //MATRIX_APPS_SYSTEM_SERVICE_H_
