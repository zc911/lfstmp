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
  int timeout = 5;
public:
  DataClient() {

  }
  MatrixError SendBatchData(string address, vector<VehicleObj> vs, vector<PedestrianObj> ps) {
    MatrixError err;


    ::model::BatchDataRequest batchReq;
    for (int j = 0; j < vs.size(); j++) {
      VehicleObj v = vs[j];
      for (int i = 0; i < v.vehicle_size(); i++) {
        string bindata = "";
        ::model::ObjType typeValue = model::UNKNOWNOBJ;
        dg::model::RecVehicle *mV = v.mutable_vehicle(i);
        if (mV->vehicletype() == dg::model::OBJ_TYPE_CAR) {
          model::Vehicle pbVehicle;
          vehicle2Protobuf(pbVehicle, mV, v.metadata());
          bindata = pbVehicle.SerializeAsString();
          typeValue = model::VEHICLE;
        } else if (mV->vehicletype() == dg::model::OBJ_TYPE_BICYCLE) {
          model::Bicycle pbBicycle;
          bicycle2Protobuf(pbBicycle, mV, v.metadata());
          bindata = pbBicycle.SerializeAsString();
          typeValue = model::BICYCLE;
        } else if (mV->vehicletype() == dg::model::OBJ_TYPE_TRICYCLE) {
          model::Tricycle pbTricycle;
          tricycle2Protobuf(pbTricycle, mV, v.metadata());
          bindata = pbTricycle.SerializeAsString();
          typeValue = model::TRICYCLE;
        } else if (mV->vehicletype() == dg::model::OBJ_TYPE_PEDESTRIAN) {
          model::Pedestrian pbPedestrian;
          pedestrian2Protobuf(pbPedestrian, mV, v.metadata());
          bindata = pbPedestrian.SerializeAsString();
          typeValue = model::PEDESTRIAN;
        }
        if (typeValue != model::UNKNOWNOBJ) {
          model::GenericObj *genObj = batchReq.add_entities();
          genObj->set_fmttype(model::PROTOBUF);
          genObj->set_type(typeValue);
          genObj->set_bindata(bindata);
        }
      }
    }
    for (int j = 0; j < ps.size(); j++) {
      PedestrianObj p = ps[j];

      for (int i = 0; i < p.pedestrian_size(); i++) {
        dg::model::RecPedestrian *mP = p.mutable_pedestrian(i);
        model::Pedestrian pbPedestrian;
        pedestrian2Protobuf(pbPedestrian, mP, p.metadata());
        string bindata = pbPedestrian.SerializeAsString();
        model::GenericObj *genObj = batchReq.add_entities();
        genObj->set_fmttype(model::PROTOBUF);
        genObj->set_type(model::PEDESTRIAN);
        genObj->set_bindata(bindata);
      }
    }
    unique_lock<mutex> lock(mtx);
    LOG(INFO)<<address;
    map<string, std::unique_ptr<DataService::Stub> >::iterator it = stubs_.find(address);
    if (it == stubs_.end()) {
      if (-1 == CreateConnect(address)) {
        err.set_code(-1);
        return err;
      }
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
    lock.unlock();

    void *got_tag;
    bool ok = false;
    cq.Next(&got_tag, &ok);

    if (status.ok()) {
      VLOG(VLOG_SERVICE) << "send to postgres success" << endl;

      return err;
    } else {
      VLOG(VLOG_SERVICE) << "send to postgres failed " << status.error_code() << endl;
      unique_lock<mutex> lock(mtx);

      map<string, std::unique_ptr<DataService::Stub> >::iterator it;
      if ((it = stubs_.find(address)) != stubs_.end())
        stubs_.erase(it);
      lock.unlock();

      return err;
    }

  }
  int CreateConnect(string address) {
    shared_ptr<grpc::Channel> channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());

    std::unique_ptr<DataService::Stub> stub(DataService::NewStub(channel));
    stubs_.insert(std::make_pair(address, std::move(stub)));
    if (stubs_.size() > 10) {
      unique_lock<mutex> lock(mtx);

      stubs_.erase(stubs_.begin());
      lock.unlock();

    }
    return 1;

  };
private:
    std::mutex mtx;
    void pedestrian2Protobuf(model::Pedestrian &pbPedestrian, dg::model::RecPedestrian *recPedestrian, const dg::model::SrcMetadata &srcMetadata) {
        model::VideoMetadata *metadata = pbPedestrian.mutable_metadata();
        model::CutboardImage *mCutImage = pbPedestrian.mutable_img();

        metadata->set_timestamp(srcMetadata.timestamp());
        metadata->set_sensorurl(srcMetadata.sensorurl());

        mCutImage->mutable_cutboard()->set_x((int)recPedestrian->img().cutboard().x() > 0 ? recPedestrian->img().cutboard().x() : 0);
        mCutImage->mutable_cutboard()->set_y((int)recPedestrian->img().cutboard().y() > 0 ? recPedestrian->img().cutboard().y() : 0);
        mCutImage->mutable_cutboard()->set_width(recPedestrian->img().cutboard().width());
        mCutImage->mutable_cutboard()->set_height(recPedestrian->img().cutboard().height());
        mCutImage->mutable_cutboard()->set_reswidth(recPedestrian->img().cutboard().reswidth());
        mCutImage->mutable_cutboard()->set_resheight(recPedestrian->img().cutboard().resheight());

        model::Image *image = mCutImage->mutable_img();
        image->set_bindata(recPedestrian->img().img().bindata());

        pbPedestrian.set_age(recPedestrian->pedesattr().age().id());
        if (recPedestrian->pedesattr().national().confidence() < 0.5) {
            pbPedestrian.set_gender(1);
        } else {
            pbPedestrian.set_gender(0);
        }
        if (recPedestrian->pedesattr().national().confidence() > 0.5) {
            pbPedestrian.set_ethnic(1);
        } else {
            pbPedestrian.set_ethnic(0);
        }

        unsigned int featuresTmp = 0;
        unsigned int headsTmp = 0;
        unsigned int upperColorsTmp = 0;
        unsigned int lowerColorsTmp = 0;
        unsigned int ageTmp = 0;
        unsigned int upperStyleTmp = 0;
        unsigned int lowerStyleTmp = 0;
        unsigned int genderTmp = 0;
        unsigned int ethnicTmp = 0;

        for (int j = 0; j < recPedestrian->pedesattr().category_size(); ++j) {
            CategoryAndFeature caf = recPedestrian->pedesattr().category(j);
            if (caf.id() == 0) {
                // Heads
                for (int k = 0; k < caf.items_size(); ++k) {
                   headsTmp |= 1 << (caf.items(k).id() - 6);
                }
                pbPedestrian.set_heads(headsTmp);
            } else if (caf.id() == 1) {
                // Features
                for (int k = 0; k < caf.items_size(); ++k) { 
                    featuresTmp |= 1 << caf.items(k).id();
                }
                pbPedestrian.set_features(featuresTmp);
            } else if (caf.id() == 2) {
              // UpperColors
                for (int k = 0; k < caf.items_size(); ++k) { 
                    upperColorsTmp |= 1 << (caf.items(k).id() - 10);
                }
              pbPedestrian.set_uppercolors(upperColorsTmp);
            } else if (caf.id() == 3) {
              // UpperStyle 
                pbPedestrian.set_upperstyle(caf.items(0).id());
                for (int k = 1; k < caf.items_size(); ++k) { 
                    if (caf.items(pbPedestrian.upperstyle()).confidence() < caf.items(k).confidence()) { 
                        pbPedestrian.set_upperstyle(caf.items(k).id()); 
                    }
                }
            } else if (caf.id() == 5) {
              // LowerColors
                for (int k = 0; k < caf.items_size(); ++k) { 
                    lowerColorsTmp |= 1 << (caf.items(k).id() - 22); 
                } 
                pbPedestrian.set_lowercolors(lowerColorsTmp);
            } else if (caf.id() == 7) {
              // LowerSytle
                pbPedestrian.set_lowerstyle(caf.items(0).id());
                for (int k = 1; k < caf.items_size(); ++k) {
                    if (caf.items(pbPedestrian.lowerstyle()).confidence() < caf.items(k).confidence()) { 
                        pbPedestrian.set_lowerstyle(caf.items(k).id());
                    }
                }
            }
        }

      /*
              unsigned int featuresTmp = 0;
              unsigned int headsTmp = 0;
              unsigned int upperColorsTmp = 0;
              unsigned int lowerColorsTmp = 0;
              unsigned int ageTmp = 0;
              unsigned int upperStyleTmp = 0;
              unsigned int lowerStyleTmp = 0;
              unsigned int genderTmp = 0;
              unsigned int ethnicTmp = 0;
              float age_conf = 0.0;
              float upper_conf = 0.0;
              float lower_conf = 0.0;
              float sex_conf = 0.0;
              float ethnic_conf = 0.0;
              for (size_t i = 0; i < recVehicle->pedestrianattrs_size(); i++) {
                  if (i >= 0 && i < 6) {
                      featuresTmp |= 1 << recVehicle->pedestrianattrs(i).attrid();
                  } else if (i >= 6 && i < 10) {
                      headsTmp |= 1 << (recVehicle->pedestrianattrs(i).attrid() - 6);
                  } else if (i >= 10 && i < 22) {
                      upperColorsTmp |= 1 << (recVehicle->pedestrianattrs(i).attrid() - 10);
                  } else if (i >= 22 && i < 34) {
                      lowerColorsTmp |= 1 << (recVehicle->pedestrianattrs(i).attrid() - 22);
                  } else if (i >= 34 && i < 38) {
                      if (recVehicle->pedestrianattrs(i).confidence() > age_conf) {
                          pbPedestrian.set_age(recVehicle->pedestrianattrs(i).attrid());
                      }
                  } else if (i >= 38 && i < 42) {
                      if (recVehicle->pedestrianattrs(i).confidence() > upper_conf) {
                          pbPedestrian.set_upperstyle(recVehicle->pedestrianattrs(i).attrid());
                      }
                  } else if (i >= 42 && i < 45) {
                      if (recVehicle->pedestrianattrs(i).confidence() > lower_conf) {
                          pbPedestrian.set_lowerstyle(recVehicle->pedestrianattrs(i).attrid());
                      }
                  } else if (i == 45) {
                      if (recVehicle->pedestrianattrs(i).confidence() > sex_conf) {
                          pbPedestrian.set_gender(1);
                      } else {
                          pbPedestrian.set_gender(0);
                      }
                  } else if (i <= 46) {
                      if (recVehicle->pedestrianattrs(i).confidence() > ethnic_conf) {
                          pbPedestrian.set_ethnic(1);
                      } else {
                          pbPedestrian.set_ethnic(0);
                      }
                  }
                      **/
  }
  void bicycle2Protobuf(model::Bicycle &pbBicycle, dg::model::RecVehicle *recVehicle, const dg::model::SrcMetadata &srcMetadata) {
    model::VideoMetadata *metadata = pbBicycle.mutable_metadata();
    model::Color *mColor = pbBicycle.mutable_color();
    model::CutboardImage *mCutImage = pbBicycle.mutable_img();

    mColor->set_id(recVehicle->color().colorid());
    mColor->set_confidence(recVehicle->color().confidence());

    metadata->set_timestamp(srcMetadata.timestamp());
    metadata->set_sensorurl(srcMetadata.sensorurl());

    mCutImage->mutable_cutboard()->set_x((int)recVehicle->img().cutboard().x() > 0 ? recVehicle->img().cutboard().x() : 0);
    mCutImage->mutable_cutboard()->set_y((int)recVehicle->img().cutboard().y() > 0 ? recVehicle->img().cutboard().y() : 0);
    mCutImage->mutable_cutboard()->set_width(recVehicle->img().cutboard().width());
    mCutImage->mutable_cutboard()->set_height(recVehicle->img().cutboard().height());
    mCutImage->mutable_cutboard()->set_reswidth(recVehicle->img().cutboard().reswidth());
    mCutImage->mutable_cutboard()->set_resheight(recVehicle->img().cutboard().resheight());

    model::Image *image = mCutImage->mutable_img();
    image->set_bindata(recVehicle->img().img().bindata());
    pbBicycle.set_feature(recVehicle->features());
  }
  void tricycle2Protobuf(model::Tricycle &pbTricycle, dg::model::RecVehicle *recVehicle, const dg::model::SrcMetadata &srcMetadata) {
    model::VideoMetadata *metadata = pbTricycle.mutable_metadata();
    model::Color *mColor = pbTricycle.mutable_color();
    model::CutboardImage *mCutImage = pbTricycle.mutable_img();

    mColor->set_id(recVehicle->color().colorid());
    mColor->set_confidence(recVehicle->color().confidence());

    metadata->set_timestamp(srcMetadata.timestamp());
    metadata->set_sensorurl(srcMetadata.sensorurl());

    mCutImage->mutable_cutboard()->set_x((int)recVehicle->img().cutboard().x() > 0 ? recVehicle->img().cutboard().x() : 0);
    mCutImage->mutable_cutboard()->set_y((int)recVehicle->img().cutboard().y() > 0 ? recVehicle->img().cutboard().y() : 0);
    mCutImage->mutable_cutboard()->set_width(recVehicle->img().cutboard().width());
    mCutImage->mutable_cutboard()->set_height(recVehicle->img().cutboard().height());
    mCutImage->mutable_cutboard()->set_reswidth(recVehicle->img().cutboard().reswidth());
    mCutImage->mutable_cutboard()->set_resheight(recVehicle->img().cutboard().resheight());
    model::Image *image = mCutImage->mutable_img();
    image->set_bindata(recVehicle->img().img().bindata());

    pbTricycle.set_feature(recVehicle->features());
  }
  void pedestrian2Protobuf(model::Pedestrian &pbPedestrian, dg::model::RecVehicle *recVehicle, const dg::model::SrcMetadata &srcMetadata) {
    /*      model::VideoMetadata *metadata = pbPedestrian.mutable_metadata();
          model::CutboardImage *mCutImage = pbPedestrian.mutable_img();

          metadata->set_timestamp(srcMetadata.timestamp());
          metadata->set_sensorurl(srcMetadata.sensorurl());

          mCutImage->mutable_cutboard()->set_x((int)recVehicle->img().cutboard().x() > 0 ? recVehicle->img().cutboard().x() : 0);
          mCutImage->mutable_cutboard()->set_y((int)recVehicle->img().cutboard().y() > 0 ? recVehicle->img().cutboard().y() : 0);
          mCutImage->mutable_cutboard()->set_width(recVehicle->img().cutboard().width());
          mCutImage->mutable_cutboard()->set_height(recVehicle->img().cutboard().height());
          mCutImage->mutable_cutboard()->set_reswidth(recVehicle->img().cutboard().reswidth());
          mCutImage->mutable_cutboard()->set_resheight(recVehicle->img().cutboard().resheight());
          model::Image *image = mCutImage->mutable_img();
          image->set_bindata(recVehicle->img().img().bindata());

          unsigned int featuresTmp = 0;
          unsigned int headsTmp = 0;
          unsigned int upperColorsTmp = 0;
          unsigned int lowerColorsTmp = 0;
          unsigned int ageTmp = 0;
          unsigned int upperStyleTmp = 0;
          unsigned int lowerStyleTmp = 0;
          unsigned int genderTmp = 0;
          unsigned int ethnicTmp = 0;
          float age_conf = 0.0;
          float upper_conf = 0.0;
          float lower_conf = 0.0;
          float sex_conf = 0.0;
          float ethnic_conf = 0.0;
          for (size_t i = 0; i < recVehicle->pedestrianattrs_size(); i++) {
              if (i >= 0 && i < 6) {
                  featuresTmp |= 1 << recVehicle->pedestrianattrs(i).attrid();
              } else if (i >= 6 && i < 10) {
                  headsTmp |= 1 << (recVehicle->pedestrianattrs(i).attrid() - 6);
              } else if (i >= 10 && i < 22) {
                  upperColorsTmp |= 1 << (recVehicle->pedestrianattrs(i).attrid() - 10);
              } else if (i >= 22 && i < 34) {
                  lowerColorsTmp |= 1 << (recVehicle->pedestrianattrs(i).attrid() - 22);
              } else if (i >= 34 && i < 38) {
                  if (recVehicle->pedestrianattrs(i).confidence() > age_conf) {
                      pbPedestrian.set_age(recVehicle->pedestrianattrs(i).attrid());
                  }
              } else if (i >= 38 && i < 42) {
                  if (recVehicle->pedestrianattrs(i).confidence() > upper_conf) {
                      pbPedestrian.set_upperstyle(recVehicle->pedestrianattrs(i).attrid());
                  }
              } else if (i >= 42 && i < 45) {
                  if (recVehicle->pedestrianattrs(i).confidence() > lower_conf) {
                      pbPedestrian.set_lowerstyle(recVehicle->pedestrianattrs(i).attrid());
                  }
              } else if (i == 45) {
                  if (recVehicle->pedestrianattrs(i).confidence() > sex_conf) {
                      pbPedestrian.set_gender(1);
                  } else {
                      pbPedestrian.set_gender(0);
                  }
              } else if (i <= 46) {
                  if (recVehicle->pedestrianattrs(i).confidence() > ethnic_conf) {
                      pbPedestrian.set_ethnic(1);
                  } else {
                      pbPedestrian.set_ethnic(0);
                  }
              }
          }
    */
  }
  void vehicle2Protobuf(model::Vehicle &pbVehicle, dg::model::RecVehicle *recVehicle, const dg::model::SrcMetadata &srcMetadata) {
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
    if (recVehicle->modeltype().confidence() + 1 <= 0.0001) {
      mModelType->set_confidence(0);
    } else {
      mModelType->set_confidence(recVehicle->modeltype().confidence());
    }

    mPlate->set_type(recVehicle->plate().typeid_());
    mPlate->set_confidence(recVehicle->plate().confidence());
    model::Color *mPlateColor = mPlate->mutable_color();
    mPlateColor->set_id(recVehicle->plate().color().colorid());
    mPlateColor->set_confidence(recVehicle->plate().color().confidence());
    mPlate->set_platetext(recVehicle->plate().platetext());
    model::Cutboard *mPlateCutboard = mPlate->mutable_cutboard();
    mPlateCutboard->set_x(((int)recVehicle->plate().cutboard().x()) > 0 ? recVehicle->plate().cutboard().x() : 0);
    mPlateCutboard->set_y(((int)recVehicle->plate().cutboard().y()) > 0 ? recVehicle->plate().cutboard().y() : 0);
    mPlateCutboard->set_width(recVehicle->plate().cutboard().width());
    mPlateCutboard->set_height(recVehicle->plate().cutboard().height());

    mCutImage->mutable_cutboard()->set_x((int)recVehicle->img().cutboard().x() > 0 ? recVehicle->img().cutboard().x() : 0);
    mCutImage->mutable_cutboard()->set_y((int)recVehicle->img().cutboard().y() > 0 ? recVehicle->img().cutboard().y() : 0);
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
