/*============================================================================
 * File Name   : witness_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description :
 * ==========================================================================*/

#include <fstream>
#include <opencv2/core/core.hpp>
#include "pbjson/pbjson.hpp"
#include <boost/algorithm/string/split.hpp>
#include <glog/logging.h>
#include <sys/time.h>
#include <matrix_engine/model/model.h>
#include <google/protobuf/text_format.h>
#include "codec/base64.h"
#include "debug_util.h"
#include "witness_service.h"
#include "image_service.h"
//
using namespace std;
namespace dg {

static int SHIFT_COLOR = 1000;
string nofilter_flag = "";
WitnessAppsService::WitnessAppsService(Config *config, string name, int baseId)
    : config_(config),
      id_(0),
      base_id_(baseId),
      name_(name) {
    enableStorage_ = (bool) config_->Value(STORAGE_ENABLED);
    storage_address_ = (string) config_->Value(STORAGE_ADDRESS);
    enable_cutboard_ = (bool) config_->Value("EnableCutboard");

    RepoService::GetInstance().Init(*config);
}

WitnessAppsService::~WitnessAppsService() {

}


Operation WitnessAppsService::getOperation(const WitnessRequestContext &ctx) {
    Operation op;
    int type = ctx.type();
    for (int i = 0; i < ctx.functions_size(); i++) {
        switch (ctx.functions(i)) {
            case RECFUNC_NONE:
                op.Set(OPERATION_NONE);
                break;
            case RECFUNC_VEHICLE:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE);
                break;
            case RECFUNC_VEHICLE_DETECT:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_DETECT);
                break;
            case RECFUNC_VEHICLE_TRACK:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_TRACK);
                break;
            case RECFUNC_VEHICLE_STYLE:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_STYLE);
                break;
            case RECFUNC_VEHICLE_COLOR:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_COLOR);
                break;
            case RECFUNC_VEHICLE_MARKER:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_MARKER);
                break;
            case RECFUNC_VEHICLE_PLATE:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_PLATE);
                break;
            case RECFUNC_VEHICLE_FEATURE_VECTOR:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_FEATURE_VECTOR);
                break;
            case RECFUNC_VEHICLE_PEDESTRIAN_ATTR:
                if ((type == REC_TYPE_VEHICLE) || (type == REC_TYPE_ALL))
                    op.Set(OPERATION_VEHICLE_PEDESTRIAN_ATTR);
                break;
            case RECFUNC_FACE:
                if ((type == REC_TYPE_FACE) || (type == REC_TYPE_ALL) || (type == REC_TYPE_DEFAULT))
                    op.Set(OPERATION_FACE);
                break;
            case RECFUNC_FACE_DETECTOR:
                if ((type == REC_TYPE_FACE) || (type == REC_TYPE_ALL) || (type == REC_TYPE_DEFAULT))
                    op.Set(OPERATION_FACE_DETECTOR);
                break;
            case RECFUNC_FACE_FEATURE_VECTOR:
                if ((type == REC_TYPE_FACE) || (type == REC_TYPE_ALL) || (type == REC_TYPE_DEFAULT))
                    op.Set(OPERATION_FACE_FEATURE_VECTOR);
                break;
            default:
                break;
        }
    }

    return op;
}


MatrixError WitnessAppsService::getRecognizedPedestrian(const Pedestrian *pobj,
                                                        RecPedestrian *prec) {
    MatrixError err;
    const Detection &d = pobj->detection();
    std::vector<Pedestrian::Attr> attrs = pobj->attrs();

    prec->set_id(pobj->id());
    prec->set_confidence((float) pobj->confidence());

    RepoService::CopyCutboard(d, prec->mutable_img()->mutable_cutboard());

    PedestrianAttr* attr = prec->mutable_pedesattr();

    for (int i = 0; i < attrs.size(); i++) {
        // sex judge
        if(i == 45) {
            NameAndConfidence* nac = attr->mutable_sex();
            if (attrs[i].confidence < attrs[i].threshold_lower) {
                nac->set_name("男");
            }
            else if (attrs[i].confidence > attrs[i].threshold_upper){
                nac->set_name("女");
            }
            else {
                nac->set_name("未知");
            }
            nac->set_confidence(attrs[i].confidence);
        }

        // national judge
        if(i == 46) {
            NameAndConfidence *nac = attr->mutable_national();
            nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
            nac->set_confidence(attrs[i].confidence);
        }

        // age judge
        if (i >= 35 && i <= 37) {
            NameAndConfidence* nac = attr->mutable_age();
            float confidence = nac->confidence();
            if (attrs[i].confidence > confidence) {
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
        }
        // head wears judge
        if(i >= 6 && i <= 9) {
            if(attrs[i].confidence > attrs[i].threshold_upper || strcmp(nofilter_flag.c_str(), "true") == 0) {
                NameAndConfidence* nac = attr->add_headwears();
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
        }
        // body wear
        if(i >= 0 && i <= 5) {    
            if(attrs[i].confidence > attrs[i].threshold_upper || strcmp(nofilter_flag.c_str(), "true") == 0) {
                NameAndConfidence* nac = attr->add_bodywears();
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
        }
        // upper wear
        if(i >= 38 && i <= 41) {    
            if(attrs[i].confidence > attrs[i].threshold_upper || strcmp(nofilter_flag.c_str(), "true") == 0) {
                NameAndConfidence* nac = attr->add_upperwears();
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
        }
        // lower wear
        if(i >= 42 && i <= 44) {    
            if(attrs[i].confidence > attrs[i].threshold_upper || strcmp(nofilter_flag.c_str(), "true") == 0) {
                NameAndConfidence* nac = attr->add_lowerwears();
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
        }
        // upper color
        if(i >= 10 && i <= 21) {    
            if(attrs[i].confidence > attrs[i].threshold_upper || strcmp(nofilter_flag.c_str(), "true") == 0) {
                NameAndConfidence* nac = attr->add_uppercolors();
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
        }
        // lower color
        if(i >= 22 && i <= 33) {    
            if(attrs[i].confidence > attrs[i].threshold_upper || strcmp(nofilter_flag.c_str(), "true") == 0) {
                NameAndConfidence* nac = attr->add_lowercolors();
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
        }
    }

    return err;
}

MatrixError WitnessAppsService::getRecognizedVehicle(const Vehicle *vobj,
                                                     RecVehicle *vrec) {
    MatrixError err;
    vrec->set_features(vobj->feature().Serialize());


    const Detection &d = vobj->detection();

    RepoService::CopyCutboard(d, vrec->mutable_img()->mutable_cutboard());
    err = RepoService::GetInstance().FillModel(*vobj, vrec);
    vrec->mutable_modeltype()->set_confidence(vobj->confidence());
    if (err.code() < 0)
        return err;
    err = RepoService::GetInstance().FillColor(vobj->color(), vrec->mutable_color());
    if (err.code() < 0)
        return err;

    err = RepoService::GetInstance().FillPlates(vobj->plates(), vrec);
    if (err.code() < 0)
        return err;

    err = RepoService::GetInstance().FillSymbols(vobj->children(), vrec);
    if (err.code() < 0)
        return err;

    return err;
}

MatrixError WitnessAppsService::getRecognizedFace(const vector<const Face *> faceVector,
                                                  ::google::protobuf::RepeatedPtrField< ::dg::model::RecPedestrian >* recPedestrian) {
    MatrixError err;
    recPedestrian->mutable_data();
    for (int i = 0; i < faceVector.size(); ++i) {
        const Face * fobj = faceVector[i];
        struct result {
            RecPedestrian* recP;
            double coincidence;
            double distance;
        };
        auto findCandidate = [=](RecPedestrian *recP) {
            result info;
            info.recP = recP;

            int faceX = fobj->detection().box.x;
            int faceY = fobj->detection().box.y;
            int faceWidth = fobj->detection().box.width;
            int faceHeight = fobj->detection().box.height;

            int pX = recP->img().cutboard().x();
            int pY = recP->img().cutboard().y();
            int pWidth = recP->img().cutboard().width();
            int pHeight = recP->img().cutboard().height();
            int totalArea = faceWidth * faceHeight, coinArea = 0;

            if (faceX >= pX && faceY >= pY && faceX <= pX + pWidth && faceY <= pY + pHeight) {
                coinArea = min(pX + pWidth - faceX, faceWidth) * min(pY + pHeight - faceY, faceHeight);
            } else if (faceX < pX && faceY < pY && faceX + faceWidth > pX && faceY + faceHeight > pY) {
                coinArea = min(faceX + faceWidth - pX, pWidth) * min(faceY + faceHeight - pY, pHeight);
            } else if (faceX < pX && faceY > pY && faceY <= pY + pHeight && faceX + faceWidth > pX) {
                coinArea = min(faceX + faceWidth - pX, pWidth) * min(faceHeight, pY + pHeight - faceY);
            } else if (faceX > pX && faceX <= pX + pWidth && faceY < pY && faceY + faceHeight > pY) {
                coinArea = min(faceWidth, pX + pWidth - faceX) * min(faceY + faceHeight - pY, faceWidth);
            }
            info.coincidence = (double)coinArea * 100.0 / totalArea;
            info.distance = abs(faceX - pX) + abs(faceY - pY);

            return info;
        };
        vector<result> candidates;
        for (int j = 0; j < recPedestrian->size(); ++j) {
            if (!recPedestrian->Mutable(j)->has_face()) {
                candidates.push_back(findCandidate(recPedestrian->Mutable(j)));
            }
        }
        RecPedestrian* MatchedPedestrian = NULL;
        if (candidates.empty()) {
            MatchedPedestrian = recPedestrian->Add();
        } else {
            result MatchedResult = candidates[0];
            for (int j = 1; j < candidates.size(); ++j) {
                if (candidates[j].coincidence > MatchedResult.coincidence) {
                    MatchedResult = candidates[j];
                } else if (abs(candidates[j].coincidence - MatchedResult.coincidence) < 0.1) {
                    if (candidates[j].distance < MatchedResult.distance) {
                        MatchedResult = candidates[j];
                    }
                }
            }
            MatchedPedestrian = MatchedResult.recP;
        }
        RecFace* face = MatchedPedestrian->mutable_face();
        face->set_id(fobj->id());
        face->set_confidence((float) fobj->confidence());
        face->set_features(fobj->feature().Serialize());
        const Detection &d = fobj->detection();
        RepoService::CopyCutboard(d, face->mutable_img()->mutable_cutboard());
    }

    return err;
}

MatrixError WitnessAppsService::getRecognizeResult(Frame *frame,
                                                   WitnessResult *result) {
    MatrixError err;
    vector<const Face *> FacesVector;

    for (const Object *object : frame->objects()) {
        DLOG(INFO) << "recognized object: " << object->id() << ", type: " << object->type();
        switch (object->type()) {
            case OBJECT_CAR:
            case OBJECT_BICYCLE:
            case OBJECT_TRICYCLE:

                err = getRecognizedVehicle((Vehicle *) object, result->add_vehicles());
                break;
            case OBJECT_PEDESTRIAN:
                err = getRecognizedPedestrian((Pedestrian *) object, result->add_pedestrian());
                break;
            case OBJECT_FACE:
                FacesVector.push_back((Face *) object);
                break;
            default:
                LOG(WARNING) << "unknown object type: " << object->type();
                break;
        }

        if (err.code() < 0) {
            break;
        }
    }

    err =  getRecognizedFace(FacesVector, result->mutable_pedestrian());

    return err;
}

MatrixError WitnessAppsService::checkWitnessImage(const WitnessImage &wImage) {
    MatrixError err;
    if (wImage.data().uri().size() == 0
        && wImage.data().bindata().size() == 0) {
        LOG(ERROR) << "image uri and bindata are empty both";
        err.set_code(-1);
        err.set_message("image uri and bindata are empty both");
        return err;
    }
    if (wImage.data().uri().size() > 0) {
        string pre = findPrefix(wImage.data().uri(), ':');
        if (pre != "http" && pre != "https" && pre != "ftp" && pre != "file") {
            LOG(ERROR) << "Invalid URI: " << wImage.data().uri() << endl;
            err.set_code(-1);
            err.set_message("Invalid Image URI");
            return err;
        }
    }

    return err;
}

MatrixError WitnessAppsService::checkRequest(const WitnessRequest &request) {
    MatrixError err;
    if (!request.has_image() || !request.image().has_data()) {
        LOG(ERROR) << "image descriptor does not exist";
        err.set_code(-1);
        err.set_message("image descriptor does not exist");
        return err;
    }
    err = checkWitnessImage(request.image());
    if (err.code() != 0) {
        return err;
    }
    return err;
}

MatrixError WitnessAppsService::checkRequest(
    const WitnessBatchRequest &requests) {
    MatrixError err;
    int size = requests.images().size();
    if (size == 0) {
        err.set_code(-1);
        err.set_message("Not data within batch requests");
        return err;
    }

    for (int i = 0; i < size; ++i) {
        WitnessImage image = requests.images(i);
        err = checkWitnessImage(image);
        if (err.code() != 0) {
            return err;
        }
    }

    return err;
}

void storage(Frame *frame, VehicleObj *client_request_obj, string storageAddress) {
    /*   */

}

MatrixError WitnessAppsService::Recognize(const WitnessRequest *request,
                                          WitnessResponse *response) {
    VLOG(VLOG_RUNTIME_DEBUG) << "Recognize using WitnessAppsService" << name_ << endl;
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    long long timestamp = curr_time.tv_sec * 1000 + curr_time.tv_usec / 1000;
    VLOG(VLOG_SERVICE) << "Received image timestamp: " << timestamp << endl;
    const string &sessionid = request->context().sessionid();
    MatrixError err = checkRequest(*request);
    if (err.code() != 0) {
        LOG(ERROR) << "Check request failed" << endl;
        return err;
    }

    VLOG(VLOG_SERVICE) << "Get Recognize request: " << sessionid
        << ", Image URI:" << request->image().data().uri();
    VLOG(VLOG_SERVICE) << "Start processing: " << sessionid << "...";

    struct timeval start, end;
    gettimeofday(&start, NULL);
    ROIImages roiimages;

    err = ImageService::ParseImage(request->image(), roiimages);
    if (err.code() != 0) {
        LOG(ERROR) << "parse image failed, " << err.message();
        return err;
    }
    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Parse Image cost: " << TimeCostInMs(start, end) << endl;

    Identification curr_id = id_++;  //TODO: make thread safe
    Frame *frame = new Frame(curr_id, roiimages.data);
    frame->set_operation(getOperation(request->context()));
    frame->set_roi(roiimages.rois);

    FrameBatch framebatch(curr_id * 10);
    framebatch.AddFrame(frame);
    gettimeofday(&start, NULL);
    //fill srcmetadata

    if (request->image().has_witnessmetadata() && request->image().witnessmetadata().timestamp() != 0) {
        timestamp = request->image().witnessmetadata().timestamp();
    }
    // engine_.Process(&framebatch);
    MatrixEnginesPool<WitnessEngine> *engine_pool = MatrixEnginesPool<WitnessEngine>::GetInstance();

    EngineData data;
    data.func = [&framebatch, &data]() -> void {
      return (bind(&WitnessEngine::Process, (WitnessEngine *) data.apps,
                   placeholders::_1))(&framebatch);
    };

    engine_pool->enqueue(&data);

    data.Wait();


    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Rec Image cost(pure): " << TimeCostInMs(start, end) << endl;

    gettimeofday(&start, NULL);

    //fill response
    WitnessResponseContext *ctx = response->mutable_context();
    ctx->set_sessionid(sessionid);
    ctx->mutable_requestts()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_requestts()->set_nanosecs((int64_t) curr_time.tv_usec);
    ctx->set_status("200");
    ctx->set_message("SUCCESS");

    //debug information of this request
    ::google::protobuf::Map<::std::string, ::dg::Time> &debugTs = *ctx
        ->mutable_debugts();
    WitnessResult *result = response->mutable_result();
    result->mutable_image()->mutable_data()->set_uri(
        request->image().data().uri());
    result->mutable_image()->mutable_data()->set_height(frame->payload()->data().rows);
    result->mutable_image()->mutable_data()->set_width(frame->payload()->data().cols);
    
    string in = "Nofilter";
    const ::google::protobuf::Map<string, string> &Params = request->context().params();
    if(Params.find(in) != Params.end())
        nofilter_flag = Params.at(in);
    else 
        nofilter_flag = "false";
    err = getRecognizeResult(frame, result);

    if (err.code() != 0) {
        LOG(ERROR) << "get result from frame failed, " << err.message();
        return err;
    }

    gettimeofday(&end, NULL);

    // return back the image data
//    WitnessImage *ret_image = result->mutable_image();
//    ret_image->CopyFrom(request->image());
//    ret_image->mutable_data()->set_width(image.cols);
//    ret_image->mutable_data()->set_height(image.rows);
    //if ReturnsImage, compress image into data.bindata

    gettimeofday(&curr_time, NULL);
    ctx->mutable_responsets()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_responsets()->set_nanosecs((int64_t) curr_time.tv_usec);
    VLOG(VLOG_PROCESS_COST) << "Parse results cost: " << TimeCostInMs(start, end) << endl;

    if (enableStorage_) {
        string storageAddress;
        if (request->context().has_storage()) {
            storageAddress = (string) request->context().storage().address();
        } else {
            storageAddress = storage_address_;
        }

        int dbType = KAFKA;
   
        const WitnessResult &r = response->result();
        if (r.vehicles_size() != 0) {
            shared_ptr<WitnessVehicleObj> client_request_obj(new WitnessVehicleObj);
            client_request_obj->mutable_storage()->set_address(storageAddress);
            for (int i = 0; i < r.vehicles_size(); i++) {
                Cutboard c = r.vehicles(i).img().cutboard();
                Mat roi(frame->payload()->data(), Rect(c.x(), c.y(), c.width(), c.height()));
                RecVehicle *v = client_request_obj->mutable_vehicleresult()->add_vehicle();
                v->CopyFrom(r.vehicles(i));
                if (enable_cutboard_) {
                    vector<char> data(roi.datastart, roi.dataend);
                    string imgdata = Base64::Encode(data);
                    v->mutable_img()->mutable_img()->set_bindata(imgdata);
                }
            }
            VehicleObj *vehicleObj = client_request_obj->mutable_vehicleresult();
            //origin img info
            vehicleObj->mutable_img()->set_uri(request->image().data().uri());
            vehicleObj->mutable_img()->set_height(frame->payload()->data().rows);
            vehicleObj->mutable_img()->set_width(frame->payload()->data().cols);
            //src metadata
            vehicleObj->mutable_metadata()->CopyFrom(request->image().witnessmetadata());
            vehicleObj->mutable_metadata()->set_timestamp(timestamp);
            //      string s;
            //       google::protobuf::TextFormat::PrintToString(*client_request_obj.get(), &s);
            //        VLOG(VLOG_SERVICE) << s << endl;
            WitnessBucket::Instance().Push(client_request_obj);
        }
    }

    VLOG(VLOG_SERVICE) << "recognized objects: " << frame->objects().size() << endl;
    VLOG(VLOG_SERVICE) << "Finish processing: " << sessionid << "..." << endl;
    VLOG(VLOG_SERVICE) << "=======" << endl;

    return err;

}

MatrixError WitnessAppsService::BatchRecognize(
    const WitnessBatchRequest *batchRequest,
    WitnessBatchResponse *batchResponse) {


    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    MatrixError err;
    long long timestamp = curr_time.tv_sec * 1000 + curr_time.tv_usec / 1000;
    const string &sessionid = batchRequest->context().sessionid();

    const ::google::protobuf::RepeatedPtrField<::dg::model::WitnessImage> &images =
        batchRequest->images();
    Identification curr_id = base_id_ * 10000 + id_++;
    VLOG(VLOG_SERVICE)
    << "Batch recognize using " << name_ << " and batch id: "
        << curr_id << endl;
    VLOG(VLOG_SERVICE)
    << "Get Batch Recognize request: " << sessionid << ", batch size:" << images.size() << " and batch id: "
        << curr_id << endl;


    err = checkRequest(*batchRequest);
    if (err.code() != 0) {
        LOG(ERROR) << "Check request failed: " << sessionid << endl;
        return err;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);


    FrameBatch framebatch(curr_id);
    vector<WitnessImage> imgDesc;
    vector<ROIImages> roiimages;
    vector<SrcMetadata> srcMetadatas;
    vector<
        ::google::protobuf::RepeatedPtrField<
            const ::dg::model::WitnessRelativeROI> > roisr;
    vector<
        ::google::protobuf::RepeatedPtrField<
            const ::dg::model::WitnessMarginROI> > roism;

    ::google::protobuf::RepeatedPtrField<const ::dg::model::WitnessImage>::iterator itr =
        images.begin();
    while (itr != images.end()) {
        if (itr->has_witnessmetadata() && itr->witnessmetadata().timestamp() != 0) {
            timestamp = itr->witnessmetadata().timestamp();
        }
        SrcMetadata metadata;
        metadata.CopyFrom(itr->witnessmetadata());
        metadata.set_timestamp(timestamp);
        srcMetadatas.push_back(metadata);
        imgDesc.push_back(const_cast<WitnessImage &>(*itr));
        itr++;
    }

    ImageService::ParseImage(imgDesc, roiimages, 10, true);

    for (int i = 0; i < roiimages.size(); ++i) {
        ROIImages image = roiimages[i];
        Identification curr_id = id_++;  //TODO: make thread safe
        Frame *frame = new Frame(curr_id, image.data);
        frame->set_operation(getOperation(batchRequest->context()));
        frame->set_roi(image.rois);

        framebatch.AddFrame(frame);
    }

    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Parse batch Image cost: " << TimeCostInMs(start, end) << endl;

    gettimeofday(&start, NULL);

    DLOG(INFO) << "Request batch size: " << framebatch.batch_size() << endl;
    VLOG(VLOG_SERVICE) << "Start processing: " << sessionid << " and id:" << framebatch.id() << endl;
    // rec_lock_.lock();
    MatrixEnginesPool<WitnessEngine> *engine_pool = MatrixEnginesPool<WitnessEngine>::GetInstance();

    EngineData data;
    data.func = [&framebatch, &data]() -> void {
      return (bind(&WitnessEngine::Process, (WitnessEngine *) data.apps,
                   placeholders::_1))(&framebatch);
    };

    if (engine_pool == NULL) {
        LOG(ERROR) << "Engine pool not initailized. " << endl;
        return err;
    }

    engine_pool->enqueue(&data);

    data.Wait();


    // rec_lock_.unlock();
    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Rec batch Image cost(pure): " << TimeCostInMs(start, end) << endl;

    gettimeofday(&start, NULL);
    //fill response
    WitnessResponseContext *ctx = batchResponse->mutable_context();
    ctx->set_sessionid(sessionid);
    ctx->mutable_requestts()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_requestts()->set_nanosecs((int64_t) curr_time.tv_usec);
    ctx->set_status("200");
    ctx->set_message("SUCCESS");

    //debug information of this request
    ::google::protobuf::Map<::std::string, ::dg::Time> &debugTs = *ctx
        ->mutable_debugts();

    vector<Frame *> frames = framebatch.frames();
    for (int i = 0; i < frames.size(); ++i) {
        Frame *frame = frames[i];
        ::dg::model::WitnessResult *result = batchResponse->add_results();
        string uri = imgDesc[i].data().uri();
        result->mutable_image()->mutable_data()->set_uri(uri);
        result->mutable_image()->mutable_data()->set_height(frame->payload()->data().rows);
        result->mutable_image()->mutable_data()->set_width(frame->payload()->data().cols);
        err = getRecognizeResult(frame, result);
        if (err.code() != 0) {
            LOG(ERROR) << "get result from frame failed, " << err.message();
            return err;
        }
    }

    if (frames.size() != batchResponse->results().size()) {
        LOG(ERROR) << "Input frame size not equal to results size." << frames.size() << "-"
            << batchResponse->results().size() << endl;
        err.set_code(-1);
        err.set_message("Input frame size not equal to results size.");
        return err;
    }

    gettimeofday(&end, NULL);
    VLOG(VLOG_PROCESS_COST) << "Parse batch results cost: " << TimeCostInMs(start, end) << endl;

    gettimeofday(&curr_time, NULL);
    ctx->mutable_responsets()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_responsets()->set_nanosecs((int64_t) curr_time.tv_usec);


    if (enableStorage_) {
        string storageAddress;
        if (batchRequest->context().has_storage()) {
            storageAddress = (string) batchRequest->context().storage().address();
        } else {
            storageAddress = storage_address_;
        }
        int dbTypeInt = KAFKA;
           
        for (int k = 0; k < batchResponse->results_size(); k++) {
            const WitnessResult &r = batchResponse->results(k);
            if (r.vehicles_size() != 0) {

                shared_ptr<WitnessVehicleObj> client_request_obj(new WitnessVehicleObj);
                client_request_obj->mutable_storage()->set_address(storageAddress);
                for (int i = 0; i < r.vehicles_size(); i++) {
                    Cutboard c = r.vehicles(i).img().cutboard();
                    Mat roi(framebatch.frames()[k]->payload()->data(), Rect(c.x(), c.y(), c.width(), c.height()));
                    RecVehicle *v = client_request_obj->mutable_vehicleresult()->add_vehicle();
                    v->CopyFrom(r.vehicles(i));
                    if (enable_cutboard_) {
                        vector<uchar> data(roi.datastart, roi.dataend);
                        string imgdata = Base64::Encode(data);
                        v->mutable_img()->mutable_img()->set_bindata(imgdata);
                    }
                    client_request_obj->mutable_vehicleresult()->mutable_metadata()->CopyFrom(srcMetadatas[k]);
                    client_request_obj->mutable_vehicleresult()->mutable_img()->set_uri(batchRequest->images(k).data().uri());

                    client_request_obj->mutable_vehicleresult()->mutable_img()->set_height(framebatch.frames()[k]->payload()->data().rows);
                    client_request_obj->mutable_vehicleresult()->mutable_img()->set_width(framebatch.frames()[k]->payload()->data().cols);
                }
                WitnessBucket::Instance().Push(client_request_obj);
            }
        }
    }


    VLOG(VLOG_SERVICE) << "Finish batch processing: " << sessionid << "..." << endl;
    return err;
}

}

