/*============================================================================
 * File Name   : witness_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description :
 * ==========================================================================*/

#include <fstream>
#include <glog/logging.h>
#include <sys/time.h>
#include <uuid/uuid.h>
#include <opencv2/core/core.hpp>
#include <boost/algorithm/string/split.hpp>

#include <matrix_engine/model/model.h>
#include <google/protobuf/text_format.h>
#include "pbjson/pbjson.hpp"

#include "codec/base64.h"
#include "debug_util.h"
#include "witness_service.h"
#include "image_service.h"
#include "witness_bucket.h"


//
using namespace std;
namespace dg {

static int SHIFT_COLOR = 1000;
static bool nofilter_flag = false;

const static unsigned int PARSE_IMAGE_TIMEOUT_DEFAULT = 60;
WitnessAppsService::WitnessAppsService(Config *config, string name, int baseId)
    : config_(config),
      id_(0),
      base_id_(baseId),
      name_(name) {
    enable_cutboard_ = (bool) config_->Value(ENABLE_CUTBOARD);
    parse_image_timeout_ = (int) config_->Value(PARSE_IMAGE_TIMEOUT);
    parse_image_timeout_ = parse_image_timeout_ == 0 ? PARSE_IMAGE_TIMEOUT_DEFAULT : parse_image_timeout_;

    RepoService::GetInstance().Init(*config);
    enable_storage_ = (bool) config_->Value(STORAGE_ENABLED);
    fullimage_storage_address_ = (string) config_->Value(STORAGE_ADDRESS);
    int typeNum = config_->Value(STORAGE_DB_TYPE + "/Size");
    int addressNum = config_->Value(STORAGE_ADDRESS + "/Size");
    if (typeNum != addressNum) {
        enable_storage_ = false;
        return;
    }
    if (enable_storage_) {
        for (int i = 0; i < typeNum; i++) {
            int type = (int) config_->Value(STORAGE_DB_TYPE + to_string(i));
            string address = (string) config_->Value(STORAGE_ADDRESS + to_string(i));
            if (type == FILEIMAGE) {
                enable_fullimage_storage_ = true;
                fullimage_storage_address_ = address;
                continue;
            }
            StorageConfig *sc = storage_configs_.Add();
            sc->set_address(address);
            sc->set_type((DBType )type);
        }
        pool_ = new ThreadPool(1);
    }

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
    std::map<std::string, float> threshold = pobj->threshold();

    prec->set_id(pobj->id());
    prec->set_confidence((float) pobj->confidence());

    RepoService::CopyCutboard(d, prec->mutable_img()->mutable_cutboard());

    PedestrianAttr* attr = prec->mutable_pedesattr();

    for (int i = 0; i < attrs.size(); i++) {
        // sex judge
        if (i == 45) {
            NameAndConfidence *nac = attr->mutable_sex();
            if (attrs[i].confidence < attrs[i].threshold_lower) {
                nac->set_name("men");
                nac->set_confidence(1 - attrs[i].confidence);
            }
            else if (attrs[i].confidence > attrs[i].threshold_upper) {
                nac->set_name("women");
                nac->set_confidence(attrs[i].confidence);
            }
            else {
                nac->set_name("unknown");
                nac->set_confidence(attrs[i].confidence);
            }
            nac->set_id(i);

        }

        // national judge
        if (i == 46) {
            NameAndConfidence *nac = attr->mutable_national();
            if (attrs[i].confidence > attrs[i].threshold_upper) {
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
            else if (attrs[i].confidence < attrs[i].threshold_lower) {
                nac->set_name("han");
                nac->set_confidence(1.0 - attrs[i].confidence);
            }
            else {
                nac->set_name("unknown");
                nac->set_confidence(attrs[i].confidence);
            }
            nac->set_id(i);

        }

        // age judge
        if (i >= 35 && i <= 37) {
            NameAndConfidence *nac = attr->mutable_age();
            float confidence = nac->confidence();
            if (attrs[i].confidence > confidence) {
                nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                nac->set_confidence(attrs[i].confidence);
            }
            nac->set_id(i);
        }
        // head wears judge
        if (i >= 6 && i <= 9) {
            if (attrs[i].confidence > attrs[i].threshold_upper) {
                if (attrs[i].confidence > threshold["HeadWears"] || nofilter_flag == true) {
                    NameAndConfidence *nac = attr->add_headwears();
                    nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                    nac->set_confidence(attrs[i].confidence);
                    nac->set_id(i);

                }
            }

        }

        // body wear
        if (i >= 0 && i <= 5) {
            if (attrs[i].confidence > attrs[i].threshold_upper) {
                if (attrs[i].confidence > threshold["BodyWears"] || nofilter_flag == true) {
                    NameAndConfidence *nac = attr->add_bodywears();
                    nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                    nac->set_confidence(attrs[i].confidence);
                    nac->set_id(i);

                }
            }
        }

        // upper color
        if (i >= 10 && i <= 21) {
            if (attrs[i].confidence > attrs[i].threshold_upper) {
                HalfOfBodyFeature *hobf = attr->mutable_upperfeatures();
                if (attrs[i].confidence > threshold["UpperColors"] || nofilter_flag == true) {
                    NameAndConfidence *nac = hobf->add_color();
                    nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                    nac->set_confidence(attrs[i].confidence);
                    nac->set_id(i);

                }
            }
        }

        // upper stripes
        if (i >= 38 && i <= 41) {
            if (attrs[i].confidence > attrs[i].threshold_upper) {
                HalfOfBodyFeature *hobf = attr->mutable_upperfeatures();
                if (attrs[i].confidence > threshold["UpperStripes"] || nofilter_flag == true) {
                    if (!hobf->has_stripes() || attrs[i].confidence > hobf->mutable_stripes()->confidence()) {
                        NameAndConfidence *nac = hobf->mutable_stripes();
                        nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                        nac->set_confidence(attrs[i].confidence);
                        nac->set_id(i);

                    }
                }
            }
        }

        // lower color
        if (i >= 22 && i <= 33) {
            if (attrs[i].confidence > attrs[i].threshold_upper) {
                HalfOfBodyFeature *hobf = attr->mutable_lowerfeatures();
                if (attrs[i].confidence > threshold["LowerColors"] || nofilter_flag == true) {
                    NameAndConfidence *nac = hobf->add_color();
                    nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                    nac->set_confidence(attrs[i].confidence);
                    nac->set_id(i);

                }
            }
        }

        // lower catagory
        if (i >= 42 && i <= 44) {
            if (attrs[i].confidence > attrs[i].threshold_upper) {
                HalfOfBodyFeature *hobf = attr->mutable_lowerfeatures();
                if (attrs[i].confidence > threshold["LowerCatagory"] || nofilter_flag == true) {
                    if (!hobf->has_catagory() || attrs[i].confidence > hobf->mutable_catagory()->confidence()) {
                        NameAndConfidence *nac = hobf->mutable_catagory();
                        nac->set_name(RepoService::GetInstance().FindPedestrianAttrName(i));
                        nac->set_confidence(attrs[i].confidence);
                        nac->set_id(i);

                    }
                }
            }
        }
    }

    return err;
}

MatrixError WitnessAppsService::getRecognizedVehicle(const Vehicle * vobj,
        RecVehicle * vrec) {
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
        ::google::protobuf::RepeatedPtrField< ::dg::model::RecPedestrian >* recPedestrian,
        int imgWidth,
        int imgHeight) {

    MatrixError err;
    recPedestrian->mutable_data();
    for (int i = 0; i < faceVector.size(); ++i) {
        const Face * fobj = faceVector[i];
        struct result {
            RecPedestrian* recP;
            double coincidence;
            double distance;
        };
        auto findCandidate = [ = ](RecPedestrian * recP) {
            result info;
            info.recP = recP;
            info.recP = recP;
            Rect FaceBox = fobj->detection().box;
            Rect BodyBox = Rect(recP->img().cutboard().x(), recP->img().cutboard().y(),
                                recP->img().cutboard().width(), recP->img().cutboard().height());
            Rect overLap = FaceBox & BodyBox;
            info.coincidence = (double)overLap.area() * 100.0 / FaceBox.area();
            info.distance = abs(FaceBox.x - BodyBox.x) + abs(FaceBox.y - FaceBox.y);

            return info;
        };
        auto Max = [](int x, int y) {return x > y ? x : y;};
        auto Min = [](int x, int y) {return x < y ? x : y;};
        vector<result> candidates;
        for (int j = 0; j < recPedestrian->size(); ++j) {
            if (!recPedestrian->Mutable(j)->has_face()) {
                result tmp = findCandidate(recPedestrian->Mutable(j));
                if (tmp.coincidence > 20) {
                    candidates.push_back(tmp);
                }
            }
        }
        RecPedestrian* MatchedPedestrian = NULL;
        bool findMatched = false;
        if (candidates.empty()) {
            MatchedPedestrian = recPedestrian->Add();
        } else {
            findMatched = true;
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
        auto faceCutboard = face->mutable_img()->mutable_cutboard();
        auto pedCutboard = MatchedPedestrian->mutable_img()->mutable_cutboard();
        //RepoService::CopyCutboard(d, face->mutable_img()->mutable_cutboard());
        if (findMatched == false) {
            MatchedPedestrian->set_id(fobj->id());
            MatchedPedestrian->set_confidence((float)fobj->confidence());
            float leftTimes = RepoService::GetInstance().FindFaceRelativePedestrian("left");
            float rightTimes = RepoService::GetInstance().FindFaceRelativePedestrian("right");
            float topTimes = RepoService::GetInstance().FindFaceRelativePedestrian("top");
            float bottomTimes = RepoService::GetInstance().FindFaceRelativePedestrian("bottom");
            pedCutboard->set_x(Max(d.box.x - d.box.width * leftTimes, 0));
            pedCutboard->set_y(Max(d.box.y - d.box.height * topTimes, 0));
            pedCutboard->set_width(Min(d.box.width * (1.0 + leftTimes + rightTimes), imgWidth -1 - pedCutboard->x()));
            pedCutboard->set_height(Min(d.box.height * (1.0 + topTimes + bottomTimes), imgHeight -1 - pedCutboard->y()));
        }
        faceCutboard->set_x(Max(d.box.x - pedCutboard->x(), 0));
        faceCutboard->set_y(Max(d.box.y - pedCutboard->y(), 0));
        faceCutboard->set_width(Min(d.box.width, pedCutboard->x() + pedCutboard->width() - d.box.x));
        faceCutboard->set_height(Min(d.box.height, pedCutboard->y() + pedCutboard->height() - d.box.y));
        faceCutboard->set_confidence(d.confidence);
    }

    return err;
}

MatrixError WitnessAppsService::getRecognizeResult(Frame * frame,
        WitnessResult * result) {
    MatrixError err;
    vector<const Face *> FacesVector;

    for (const Object *object : frame->objects()) {
        //  DLOG(INFO) << "recognized object: " << object->id() << ", type: " << object->type();
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

    err =  getRecognizedFace(FacesVector, result->mutable_pedestrian(),
                             result->image().data().width(),
                             result->image().data().height());

    return err;
}

MatrixError WitnessAppsService::checkWitnessImage(const WitnessImage & wImage) {
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

MatrixError WitnessAppsService::checkRequest(const WitnessRequest & request) {
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
    const WitnessBatchRequest & requests) {
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

void storage(Frame * frame, VehicleObj * client_request_obj, string storageAddress) {
    /*   */

}

MatrixError WitnessAppsService::Recognize(const WitnessRequest * request,
        WitnessResponse * response) {
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
    if (enable_fullimage_storage_) {
        pool_->enqueue([&roiimages, this, timestamp]() {

            string path = this->fullimage_storage_address_ + "/" + GetLatestHour();
            string dir = "mkdir -p " + path;
            const int dir_err = system(dir.c_str());
            if (-1 == dir_err)
            {
                printf("Error creating directory!n");
            }
            string name = path + "/" + to_string(timestamp) + ".jpg";
            imwrite(name, roiimages.data);
        });
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
    //engine_.Process(&framebatch);
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

    string fileterFlag = "Nofilter";
    auto Params = request->context().params();
    if (Params.find(fileterFlag) == Params.end() || Params.at(fileterFlag) == "0")
        nofilter_flag = false;
    else
        nofilter_flag = true;
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
    //if ReturnsImage, compress image ito data.bindata

    gettimeofday(&curr_time, NULL);
    ctx->mutable_responsets()->set_seconds((int64_t) curr_time.tv_sec);
    ctx->mutable_responsets()->set_nanosecs((int64_t) curr_time.tv_usec);
    VLOG(VLOG_PROCESS_COST) << "Parse results cost: " << TimeCostInMs(start, end) << endl;
    if (enable_storage_) {
        shared_ptr<WitnessVehicleObj> client_request_obj(new WitnessVehicleObj);
        WitnessResult * result = client_request_obj->results.Add();
        result->CopyFrom(response->result());
        if (request->context().storages_size() > 0) {
            client_request_obj->storages.CopyFrom(request->context().storages());
        } else {
            client_request_obj->storages.CopyFrom(storage_configs_);
        }
        client_request_obj->imgs.push_back(framebatch.frames()[0]->payload()->data());
        SrcMetadata metadata;
        metadata.set_timestamp(timestamp);
        client_request_obj->srcMetadatas.push_back(metadata);
        WitnessBucket::Instance().Push(client_request_obj);

    }


    VLOG(VLOG_SERVICE) << "recognized objects: " << frame->objects().size() << endl;
    VLOG(VLOG_SERVICE) << "Finish processing: " << sessionid << "..." << endl;
    VLOG(VLOG_SERVICE) << "=======" << endl;

    return err;

}

MatrixError WitnessAppsService::BatchRecognize(
    const WitnessBatchRequest * batchRequest,
    WitnessBatchResponse * batchResponse) {


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
    vector <
    ::google::protobuf::RepeatedPtrField <
    const ::dg::model::WitnessRelativeROI > > roisr;
    vector <
    ::google::protobuf::RepeatedPtrField <
    const ::dg::model::WitnessMarginROI > > roism;

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

    err = ImageService::ParseImage(imgDesc, roiimages, parse_image_timeout_, true);
    if (err.code() == -1) {
        cout << "Read data error" << endl;
        return err;
    }
    if (enable_fullimage_storage_) {
    std::mutex imgmt;

        pool_->enqueue([&roiimages,&imgmt, this]() {
                      std::unique_lock<mutex> imglc(imgmt);

            struct timeval curr_time;
            gettimeofday(&curr_time, NULL);
            long long timestamp = curr_time.tv_sec * 1000 + curr_time.tv_usec / 1000;

            for (int i = 0; i < roiimages.size(); i++) {
                string path = this->fullimage_storage_address_ + "/" + GetLatestHour();
                string dir = "mkdir -p " + path;
                const int dir_err = system(dir.c_str());
                if (-1 == dir_err)
                {
                    LOG(WARNING)<<("Error creating directory!n");
                    continue;
                }
                char uuidBuff[36];
                uuid_t uuidGenerated;
                uuid_generate_random(uuidGenerated);
                uuid_unparse(uuidGenerated, uuidBuff);
                string name = path + "/" + uuidBuff + ".jpg";
                imwrite(name, roiimages[i].data);
            }

        });
    }

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
    gettimeofday(&start, NULL);

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

    if (enable_storage_) {
        shared_ptr<WitnessVehicleObj> client_request_obj(new WitnessVehicleObj);
        client_request_obj->results.CopyFrom(batchResponse->results());
        if (batchRequest->context().storages_size() > 0) {
            client_request_obj->storages.CopyFrom(batchRequest->context().storages());
        } else {
            client_request_obj->storages.CopyFrom(storage_configs_);
        }
        for (int k = 0; k < batchResponse->results_size(); k++) {
            client_request_obj->imgs.push_back(framebatch.frames()[k]->payload()->data());
        }
        client_request_obj->srcMetadatas = srcMetadatas;
            gettimeofday(&start, NULL);

        WitnessBucket::Instance().Push(client_request_obj);
            gettimeofday(&end, NULL);
            VLOG(VLOG_PROCESS_COST) << "storage cost: " << TimeCostInMs(start, end) << endl;


    }


    VLOG(VLOG_SERVICE) << "Finish batch processing: " << sessionid << "..." << endl;
    return err;
}

}


/*
for (int k = 0; k < batchResponse->results_size(); k++) {
const WitnessResult &r = batchResponse->results(k);
if (r.vehicles_size() != 0) {

if (batchRequest->context().storages_size() > 0) {
client_request_obj->mutable_storages()->CopyFrom(batchRequest->context().storages());
} else {
client_request_obj->mutable_storages()->CopyFrom(storage_configs_);
}
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
}*/

