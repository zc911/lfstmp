#ifndef SRC_CLIENTS_DATA_CLIENT_H_
#define SRC_CLIENTS_DATA_CLIENT_H_
#include "dataservice.grpc.pb.h"
#include "deepdatasingle.pb.h"
#include "deepdatasingle.grpc.pb.h"
#include "localcommon.pb.h"
#include "witness.grpc.pb.h"
#include "pbjson/pbjson.hpp"
#include <google/protobuf/text_format.h>
using ::model::DataService;
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::Status;
using grpc::CompletionQueue;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;


class DataClient {
    int timeout=5;
public:
    DataClient() {

    }
	MatrixError SendBatchData(string address, VehicleObj *v){
        ::model::BatchDataRequest batchReq;
        for(int i=0;i<v->vehicle_size();i++){
            string bindata="";
            ::model::ObjType typeValue=model::UNKNOWNOBJ;
            dg::model::RecVehicle *mV = v->mutable_vehicle(i);
            if(mV->vehicletype()==dg::model::OBJ_TYPE_CAR){
                model::Vehicle pbVehicle;
                vehicle2Protobuf(pbVehicle,mV,v->metadata());
                bindata = pbVehicle.SerializeAsString();
                typeValue=model::VEHICLE;
            }else if(mV->vehicletype()==dg::model::OBJ_TYPE_BICYCLE){
                model::Bicycle pbBicycle;
                bicycle2Protobuf(pbBicycle,mV,v->metadata());
                bindata = pbBicycle.SerializeAsString();
                typeValue=model::BICYCLE;
            }else if(mV->vehicletype()==dg::model::OBJ_TYPE_TRICYCLE){
                model::Tricycle pbTricycle;
                tricycle2Protobuf(pbTricycle,mV,v->metadata());
                bindata = pbTricycle.SerializeAsString();
                typeValue=model::TRICYCLE;
            }else if(mV->vehicletype()==dg::model::OBJ_TYPE_PEDESTRIAN){
                model::Pedestrian pbPedestrian;
                pedestrian2Protobuf(pbPedestrian,mV,v->metadata());
                bindata = pbPedestrian.SerializeAsString();
                typeValue=model::PEDESTRIAN;
            }
            if(typeValue!=model::UNKNOWNOBJ){
                model::GenericObj *genObj =batchReq.add_entities();
                genObj->set_fmttype(model::PROTOBUF);
                genObj->set_type(typeValue);
                genObj->set_bindata(bindata);
            }
        }

        map<string, std::unique_ptr<DataService::Stub> >::iterator it = stubs_.find(address);
        if (it == stubs_.end()) {
            CreateConnect(address);
        }
        ::model::DataResponse reply;
        ClientContext context;
        std::chrono::system_clock::time_point
            deadline = std::chrono::system_clock::now() + std::chrono::seconds(timeout);
        context.set_deadline(deadline);
        CompletionQueue cq;
        Status status;
        std::unique_ptr<ClientAsyncResponseReader<::model::DataResponse> > rpc(
            stubs_[address]->AsyncSendBatchData(&context, batchReq, &cq));
        rpc->Finish(&reply, &status, (void *) 1);
        void *got_tag;
        bool ok = false;
        cq.Next(&got_tag, &ok);
                MatrixError err;

        if (status.ok()) {
            VLOG(VLOG_SERVICE) << "send to storage success" << endl;

            return err;
        } else {
            VLOG(VLOG_SERVICE) << "send to storage failed " << status.error_code() << endl;
            stubs_.erase(stubs_.find(address));
            return err;
        }
	}
    void CreateConnect(string address) {
        shared_ptr<grpc::Channel> channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
        std::unique_ptr<DataService::Stub> stub(DataService::NewStub(channel));
        stubs_.insert(std::make_pair(address, std::move(stub)));
        if (stubs_.size() > 10) {
            stubs_.erase(stubs_.begin());
        }
        for (map<string, std::unique_ptr<DataService::Stub> >::iterator it = stubs_.begin(); it != stubs_.end();
             it++) {
            VLOG(VLOG_SERVICE) << it->first;
        }

    };
private:
    void bicycle2Protobuf(model::Bicycle &pbBicycle,dg::model::RecVehicle *recVehicle,const dg::model::SrcMetadata &srcMetadata){
        model::VideoMetadata *metadata = pbBicycle.mutable_metadata();
        model::Color *mColor = pbBicycle.mutable_color();
        model::CutboardImage *mCutImage = pbBicycle.mutable_img();

        mColor->set_id(recVehicle->color().colorid());
        mColor->set_confidence(recVehicle->color().confidence());

        metadata->set_timestamp(srcMetadata.timestamp());
        metadata->set_sensorurl(srcMetadata.sensorurl());

        mCutImage->mutable_cutboard()->set_x((int)recVehicle->img().cutboard().x()>0?recVehicle->img().cutboard().x():0);
        mCutImage->mutable_cutboard()->set_y((int)recVehicle->img().cutboard().y()>0?recVehicle->img().cutboard().y():0);
        mCutImage->mutable_cutboard()->set_width(recVehicle->img().cutboard().width());
        mCutImage->mutable_cutboard()->set_height(recVehicle->img().cutboard().height());
        mCutImage->mutable_cutboard()->set_reswidth(recVehicle->img().cutboard().reswidth());
        mCutImage->mutable_cutboard()->set_resheight(recVehicle->img().cutboard().resheight());

        model::Image *image = mCutImage->mutable_img();
        image->set_bindata(recVehicle->img().img().bindata());
        pbBicycle.set_feature(recVehicle->features());
    }
    void tricycle2Protobuf(model::Tricycle &pbTricycle,dg::model::RecVehicle *recVehicle,const dg::model::SrcMetadata &srcMetadata){
        model::VideoMetadata *metadata = pbTricycle.mutable_metadata();
        model::Color *mColor = pbTricycle.mutable_color();
        model::CutboardImage *mCutImage = pbTricycle.mutable_img();

        mColor->set_id(recVehicle->color().colorid());
        mColor->set_confidence(recVehicle->color().confidence());

        metadata->set_timestamp(srcMetadata.timestamp());
        metadata->set_sensorurl(srcMetadata.sensorurl());

        mCutImage->mutable_cutboard()->set_x((int)recVehicle->img().cutboard().x()>0?recVehicle->img().cutboard().x():0);
        mCutImage->mutable_cutboard()->set_y((int)recVehicle->img().cutboard().y()>0?recVehicle->img().cutboard().y():0);
        mCutImage->mutable_cutboard()->set_width(recVehicle->img().cutboard().width());
        mCutImage->mutable_cutboard()->set_height(recVehicle->img().cutboard().height());
        mCutImage->mutable_cutboard()->set_reswidth(recVehicle->img().cutboard().reswidth());
        mCutImage->mutable_cutboard()->set_resheight(recVehicle->img().cutboard().resheight());
        model::Image *image = mCutImage->mutable_img();
        image->set_bindata(recVehicle->img().img().bindata());

        pbTricycle.set_feature(recVehicle->features());
    }
    void pedestrian2Protobuf(model::Pedestrian &pbPedestrian,dg::model::RecVehicle *recVehicle,const dg::model::SrcMetadata &srcMetadata){
        model::VideoMetadata *metadata = pbPedestrian.mutable_metadata();
        model::CutboardImage *mCutImage = pbPedestrian.mutable_img();

        metadata->set_timestamp(srcMetadata.timestamp());
        metadata->set_sensorurl(srcMetadata.sensorurl());

        mCutImage->mutable_cutboard()->set_x((int)recVehicle->img().cutboard().x()>0?recVehicle->img().cutboard().x():0);
        mCutImage->mutable_cutboard()->set_y((int)recVehicle->img().cutboard().y()>0?recVehicle->img().cutboard().y():0);
        mCutImage->mutable_cutboard()->set_width(recVehicle->img().cutboard().width());
        mCutImage->mutable_cutboard()->set_height(recVehicle->img().cutboard().height());
        mCutImage->mutable_cutboard()->set_reswidth(recVehicle->img().cutboard().reswidth());
        mCutImage->mutable_cutboard()->set_resheight(recVehicle->img().cutboard().resheight());
        model::Image *image = mCutImage->mutable_img();
        image->set_bindata(recVehicle->img().img().bindata());

        unsigned int featuresTmp=0;
        unsigned int headsTmp=0;
        unsigned int upperColorsTmp=0;
        unsigned int lowerColorsTmp=0;
        unsigned int ageTmp=0;
        unsigned int upperStyleTmp=0;
        unsigned int lowerStyleTmp=0;
        unsigned int genderTmp=0;
        unsigned int ethnicTmp=0;
        float age_conf=0.0;
        float upper_conf=0.0;
        float lower_conf=0.0;
        float sex_conf=0.0;
        float ethnic_conf=0.0;
        for(size_t i=0;i<recVehicle->pedestrianattrs_size();i++){
            if(i>=0&&i<6){
               featuresTmp|=1<<recVehicle->pedestrianattrs(i).attrid();
            }else if(i>=6 && i<10){
                headsTmp |= 1<<(recVehicle->pedestrianattrs(i).attrid()-6);
            }else if(i>=10 && i<22){
                upperColorsTmp|=1<<(recVehicle->pedestrianattrs(i).attrid()-10);
            }else if(i>=22 && i<34){
                lowerColorsTmp|=1<<(recVehicle->pedestrianattrs(i).attrid()-22);
            }else if(i>=34 && i<38){
                if(recVehicle->pedestrianattrs(i).confidence()>age_conf){
                    pbPedestrian.set_age(recVehicle->pedestrianattrs(i).attrid());
                }
            }else if(i>=38 && i<42){
                if(recVehicle->pedestrianattrs(i).confidence()>upper_conf){
                    pbPedestrian.set_upperstyle(recVehicle->pedestrianattrs(i).attrid());
                }
            }else if(i>=42 && i<45){
                if(recVehicle->pedestrianattrs(i).confidence()>lower_conf){
                    pbPedestrian.set_lowerstyle(recVehicle->pedestrianattrs(i).attrid());
                }
            }else if(i==45){
                if(recVehicle->pedestrianattrs(i).confidence()>sex_conf){
                    pbPedestrian.set_gender(1);
                }else{
                    pbPedestrian.set_gender(0);
                }
            }else if(i<=46){
                if(recVehicle->pedestrianattrs(i).confidence()>ethnic_conf){
                    pbPedestrian.set_ethnic(1);
                }else{
                    pbPedestrian.set_ethnic(0);
                }
            }
        }

    }
    void vehicle2Protobuf(model::Vehicle &pbVehicle,dg::model::RecVehicle *recVehicle,const dg::model::SrcMetadata &srcMetadata){
        model::VideoMetadata *metadata = pbVehicle.mutable_metadata();
        model::Color *mColor = pbVehicle.mutable_color();
        model::VehicleModelType *mModelType = pbVehicle.mutable_modeltype();
        model::CutboardImage *mCutImage = pbVehicle.mutable_img();
        model::LicensePlate *mPlate = pbVehicle.mutable_plate();
        string *feature = pbVehicle.mutable_feature();

        mColor->set_id(recVehicle->color().colorid());
        mColor->set_confidence(recVehicle->color().confidence());

        metadata->set_timestamp(srcMetadata.timestamp());
        metadata->set_sensorurl(srcMetadata.sensorurl());

        mModelType->set_type(recVehicle->modeltype().typeid_());
        mModelType->set_brandid(recVehicle->modeltype().brandid());
        mModelType->set_subbrandid(recVehicle->modeltype().subbrandid());
        mModelType->set_modelyearid(recVehicle->modeltype().modelyearid());
        if(recVehicle->modeltype().confidence()+1<=0.0001){
            mModelType->set_confidence(0);
        }else{
            mModelType->set_confidence(recVehicle->modeltype().confidence());
        }

        mPlate->set_type(recVehicle->plate().typeid_());
        mPlate->set_confidence(recVehicle->plate().confidence());
        model::Color *mPlateColor = mPlate->mutable_color();
        mPlateColor->set_id(recVehicle->plate().color().colorid());
        mPlateColor->set_confidence(recVehicle->plate().color().confidence());
        mPlate->set_platetext(recVehicle->plate().platetext());
        model::Cutboard *mPlateCutboard = mPlate->mutable_cutboard();
        mPlateCutboard->set_x(((int)recVehicle->plate().cutboard().x())>0?recVehicle->plate().cutboard().x():0);
        mPlateCutboard->set_y(((int)recVehicle->plate().cutboard().y())>0?recVehicle->plate().cutboard().y():0);
        mPlateCutboard->set_width(recVehicle->plate().cutboard().width());
        mPlateCutboard->set_height(recVehicle->plate().cutboard().height());

        mCutImage->mutable_cutboard()->set_x((int)recVehicle->img().cutboard().x()>0?recVehicle->img().cutboard().x():0);
        mCutImage->mutable_cutboard()->set_y((int)recVehicle->img().cutboard().y()>0?recVehicle->img().cutboard().y():0);
        mCutImage->mutable_cutboard()->set_width(recVehicle->img().cutboard().width());
        mCutImage->mutable_cutboard()->set_height(recVehicle->img().cutboard().height());
        mCutImage->mutable_cutboard()->set_reswidth(recVehicle->img().cutboard().reswidth());
        mCutImage->mutable_cutboard()->set_resheight(recVehicle->img().cutboard().resheight());
        model::Image *image = mCutImage->mutable_img();
        image->set_bindata(recVehicle->img().img().bindata());

        pbVehicle.set_feature(recVehicle->features());

    }
    map<string, std::unique_ptr<DataService::Stub> > stubs_;

};

#endif 
