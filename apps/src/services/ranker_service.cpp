/*============================================================================
 * File Name   : ranker_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <glog/logging.h>
#include <sstream>
#include "ranker_service.h"
#include "codec/base64.h"
#include "image_service.h"
#include "string_util.h"
#include "io/uri_reader.h"
#include "../../../engine/src/io/rank_candidates_repo.h"
#include <sys/time.h>

namespace dg {
//const int RANKER_MAXIMUM = 10000
RankerAppsService::RankerAppsService(const Config *config, string name, int baseId)
    : name_(name),
      config_(config) {
    config_ = config;

    limits_ = min(RANKER_MAXIMUM, (int) config->Value(ADVANCED_RANKER_MAXIMUM));
}

RankerAppsService::~RankerAppsService() {

}

MatrixError RankerAppsService::RankFeature(const RankFeatureRequest *request, RankFeatureResponse *response) {

    MatrixError err;
    LOG(INFO) << "Get Ranker feature request: " << request->context().sessionid() << endl;
    struct timeval ts;
    gettimeofday(&ts, NULL);
    response->mutable_context()->mutable_requestts()->set_seconds(ts.tv_sec);
    response->mutable_context()->mutable_requestts()->set_nanosecs(ts.tv_usec * 1000);
    response->mutable_context()->set_sessionid(request->context().sessionid());
    switch (request->context().type()) {
        case dg::RANK_TYPE_FACE:
            err = getRankedFaceVector(request, response);
            LOG(INFO) << "Ranker feature finish: " << request->context().sessionid() << endl;
            break;
        default:
            LOG(ERROR) << "bad request(" << request->context().sessionid() << "), unknown rank type "
                << request->context().type();
            err.set_code(-1);
            err.set_message("bad request, unknown action");
            break;
    }

    gettimeofday(&ts, NULL);
    response->mutable_context()->mutable_responsets()->set_seconds(ts.tv_sec);
    response->mutable_context()->mutable_responsets()->set_nanosecs(ts.tv_usec * 1000);
    response->mutable_context()->set_status("200");
    response->mutable_context()->set_message("SUCCESS");
    return err;

}


MatrixError RankerAppsService::AddFeatures(const AddFeaturesRequest *request, AddFeaturesResponse *response) {

    LOG(INFO) << "Get add features request: " << request->features().size() << endl;

    struct timeval ts;
    gettimeofday(&ts, NULL);
    response->mutable_context()->mutable_requestts()->set_seconds(ts.tv_sec);
    response->mutable_context()->mutable_requestts()->set_nanosecs(ts.tv_usec * 1000);
    response->mutable_context()->set_sessionid(request->context().sessionid());

    MatrixError err;
    if (request->features().size() == 0) {
        err.set_code(-1);
        err.set_message("Add features request is empty");
        return err;
    }

    response->mutable_context()->set_sessionid(request->context().sessionid());
    FeaturesFrame frame(0);


    for (auto f:request->features()) {

        if (f.feature().feature().size() == 0) {
            err.set_code(-1);
            err.set_message("Feature vector is empty");
            return err;
        }

        FaceRankFeature feature;
        feature.Deserialize(f.feature().feature());
        feature.id_ = f.info().id();
        feature.name_ = f.info().name();
        feature.image_uri_ = f.info().uri();
        if (f.info().data().size() != 0) {
            vector<uchar> imageContent;
            Base64::Decode<uchar>(f.info().data(), imageContent);
            ImageService::DecodeDataToMat(imageContent, feature.image_);
        }

        frame.AddFeature(feature);
    }

    MatrixEnginesPool<SimpleRankEngine> *engine_pool = MatrixEnginesPool<SimpleRankEngine>::GetInstance();
    EngineData data;
    data.func = [&frame, &data]() -> void {
        return (bind(&SimpleRankEngine::AddFeatures, (SimpleRankEngine *) data.apps,
                     placeholders::_1))(&frame);
    };

    if (engine_pool == NULL) {
        LOG(ERROR) << "Engine pool not initailized. " << endl;
        return err;
    }

    engine_pool->enqueue(&data);
    data.Wait();

    gettimeofday(&ts, NULL);
    response->mutable_context()->mutable_responsets()->set_seconds(ts.tv_sec);
    response->mutable_context()->mutable_responsets()->set_nanosecs(ts.tv_usec * 1000);
    response->mutable_context()->set_status("200");
    response->mutable_context()->set_message("SUCCESS");

    return err;
}

typedef boost::tokenizer<boost::char_separator<char>> MyTokenizer;

static MatrixError genFilters(const RankRequestContext &ctx, map<string, set<string>> &stringFilters,
                              map<string, pair<int, int>> &rangeFilters) {

    for (auto paramItr = ctx.params().begin(); paramItr != ctx.params().end(); ++paramItr) {
        string key = paramItr->first;
        string value = paramItr->second;
        if (boost::algorithm::starts_with(key, "Filter")) {

            key = key.substr(6, key.size());
            value = value.substr(1, value.size() - 2);
            boost::char_separator<char> sep(",");
            MyTokenizer tok(value, sep);
            MyTokenizer::iterator col = tok.begin();
            while (col != tok.end()) {
                auto stringFilterItr = stringFilters.find(key);
                if (stringFilterItr != stringFilters.end()) {
                    stringFilterItr->second.insert(*col);
                } else {
                    set<string> filters;
                    filters.insert(*col);
                    stringFilters.insert(make_pair(key, filters));
                }
//                cout << "Filter: " << key << " : " << *col << endl;
                col++;
            }

        } else if (boost::algorithm::starts_with(key, "Range")) {
            key = key.substr(5, key.size());
            string rangeStartString = value.substr(0, value.find("-"));
            string rangeEndString = value.substr(value.find("-") + 1, value.size());
            int rangeStart = atoi(rangeStartString.c_str());
            int rangeEnd = atoi(rangeEndString.c_str());

            if (rangeStart == 0 && rangeEnd == 0) {
                LOG(ERROR) << "Range filter invalid: " << key << " : " << value;
                continue;
            }
            if (rangeStart > rangeEnd) {
                LOG(ERROR) << "Range filter invalid: " << key << " : " << value;
                continue;
            }
            cout << key << "-" << rangeStart << rangeEnd << endl;
            rangeFilters.insert(make_pair(key, make_pair(rangeStart, rangeEnd)));

//            auto rangeItemItr = rangeFilters.find(key);
//            if (rangeItemItr != rangeFilters.end()) {
//
//            } else {
//
//                rangeFilters.insert(make_pair(key, make_pair(rangeStart, rangeEnd)));
////                rangeItemItr->second.first = rangeStart;
////                rangeItemItr->second.second = rangeEnd;
//            }

//            cout << "Range: " << key << ": " << rangeStart << "-" << rangeEnd << endl;
        }
    }

    return MatrixError();

}

static bool applyFilters(const RankCandidatesItem &item, const map<string, set<string>> &valueFilters,
                         const map<string, pair<int, int>> &rangeFilters) {
    if (valueFilters.size() == 0 && rangeFilters.size() == 0) {
        return true;
    }

    // apply the string filter

    cout << "apply value filter: " << endl;
    for (auto itr = valueFilters.begin(); itr != valueFilters.end(); ++itr) {
        string filterName = itr->first;
        filterName = filterName.substr(0, 6);
        auto attributeSetItr = item.attributes_.find(filterName);
        if (attributeSetItr != item.attributes_.end()) {

            auto specAttrItr = attributeSetItr->second;

            for(auto i = specAttrItr.begin(); i != specAttrItr.end(); ++i){
                cout << "attr: " << *i;
            }
            cout << endl;
            for(auto i = itr->second.begin(); i != itr->second.end(); ++i){
                cout << "filter: " << *i;
            }
            cout << endl;

            set<string> intersectionResult;
            std::set_intersection(specAttrItr.begin(),
                                  specAttrItr.end(),
                                  itr->second.begin(),
                                  itr->second.end(),
                                  insert_iterator<set<string> >(intersectionResult, intersectionResult.begin()));
            if(intersectionResult.size() == 0){

                return false;
            }

        } else {
            return false;
        }

    }

    cout << "apply range filter: " << endl;
    for(auto itr = rangeFilters.begin(); itr != rangeFilters.end(); ++itr){
        string filterName = itr->first;
        int rangeStart = itr->second.first;
        int rangeEnd = itr->second.second;

        auto attributeSetItr = item.attributes_.find(filterName);
        if(attributeSetItr != item.attributes_.end()){
            if(attributeSetItr->second.size() != 0){
                string valueString = *(attributeSetItr->second.begin());
                int value = atoi(valueString.c_str());
                cout << "value: " << value << " range: " << rangeStart << "- " << rangeEnd << endl;
                if(value < rangeStart || value > rangeEnd)
                    return false;
            }
        }else{
            return false;
        }


    }

    return true;
}


MatrixError RankerAppsService::getFaceScoredVector(
    vector<Score> &scores, const RankFeatureRequest *request,
    RankFeatureResponse *response) {

    MatrixError err;

    if (request->feature().feature().size() == 0) {
        LOG(ERROR) << "Compared feature vector is empty" << endl;
        err.set_code(-1);
        err.set_message("Compared feature vector is empty");
        return err;
    }


//    RankRequestContext *ctx = const_cast<RankRequestContext &>(request->context());

    const RankRequestContext &ctx = request->context();
    int maxCandidates = 10;

    auto itr = ctx.params().find("MaxCandidates");
    if (itr != request->context().params().end()) {
        string mcs = itr->second;
        maxCandidates = atoi(mcs.c_str());
        maxCandidates = maxCandidates <= 0 ? 10 : maxCandidates;
//        ctx->mutable_params()->erase(itr);
    }


    bool needImageData = true;
    itr = ctx.params().find("ImageData");
    if (itr != request->context().params().end()) {
        string needImageDataString = itr->second;
        if (needImageDataString == "false") {
            needImageData = false;
        }
        if (needImageDataString == "true") {
            needImageData = true;
        }
//        ctx->mutable_params()->erase(itr);
    }

    cout << "The filter params size: " << ctx.params().size() << endl;

    map<string, set<string>> stringFilters;
    map<string, pair<int, int>> rangeFilters;
    genFilters(ctx, stringFilters, rangeFilters);

    cout << "String filter from request" << endl;
    for (auto itr = stringFilters.begin(); itr != stringFilters.end(); ++itr) {
        cout << itr->first << ": [";
        for (auto itr2 = itr->second.begin(); itr2 != itr->second.end(); ++itr2) {
            cout << *itr2 << ",";
        }
        cout << "]";
    }
    cout << endl;
    cout << "Range filter from request" << endl;
    for (auto itr3 = rangeFilters.begin(); itr3 != rangeFilters.end(); ++itr3) {
        cout << itr3->first << ": [";
        cout << itr3->second.first << "," << itr3->second.second;
        cout << "]";
    }
    cout << endl;

    FaceRankFeature feature;
    Base64::Decode(request->feature().feature(), feature.feature_);

    FaceRankFrame f(0, feature);
    f.max_candidates_ = maxCandidates * 100;
    Operation op;

    op.Set(OPERATION_FACE_FEATURE_VECTOR);

    f.set_operation(op);
    MatrixEnginesPool<SimpleRankEngine> *engine_pool = MatrixEnginesPool<SimpleRankEngine>::GetInstance();
    EngineData data;
    data.func = [&f, &data]() -> void {
        return (bind(&SimpleRankEngine::RankFace, (SimpleRankEngine *) data.apps,
                     placeholders::_1))(&f);
    };

    if (engine_pool == NULL) {
        LOG(ERROR) << "Engine pool not initailized. " << endl;
        return err;
    }

    engine_pool->enqueue(&data);
    data.Wait();

    RankCandidatesRepo &repo = RankCandidatesRepo::GetInstance();
    for (auto r : f.result_) {


        const RankCandidatesItem &item = repo.Get(r.index_);

        if(!applyFilters(item, stringFilters, rangeFilters)){
            cout << "Filter failed: " << item.id_ << "," << item.name_ << endl;
            continue;
        }

        VLOG(VLOG_RUNTIME_DEBUG) << r.index_ << " " << r.score_ << "" << item.id_ << " " << item.image_uri_ << endl;

        RankItem *result = response->mutable_candidates()->Add();
        result->set_uri(item.image_uri_);
        result->set_id(item.id_);
        result->set_score(r.score_);
        map<string, set<string>> attrs = item.attributes_;
        for(auto itr = attrs.begin(); itr != attrs.end(); ++itr){
            std::stringstream ss;
            for(auto itr2 = itr->second.begin(); itr2 != itr->second.end(); ++itr2){
                if(itr2 != itr->second.begin()){
                    ss << ",";
                }
                ss << *itr2;
            }
            (*result->mutable_attributes())[itr->first] = ss.str();
        }


        vector<uchar> imageContent;
        if (needImageData) {

            if ((item.image_.cols & item.image_.rows) != 0) {
                result->set_data(encode2JPEGInBase64(item.image_));
            } else {
                try {
                    if (UriReader::Read(item.image_uri_, imageContent, 3) >= 0) {
                        result->set_data(Base64::Encode<uchar>(imageContent));
                    }
                } catch (exception &e) {
                    LOG(ERROR) << "Uri read failed: " << item.image_uri_ << endl;
                }
            }
        }


        result->set_name(item.name_);
    }


    return err;
}

MatrixError RankerAppsService::getRankedFaceVector(
    const RankFeatureRequest *request,
    RankFeatureResponse *response) {

    MatrixError err;

    vector<Score> scores;
    err = getFaceScoredVector(scores, request, response);

    return err;
}

//
//MatrixError RankerAppsService::GetRankedVector(
//    const FeatureRankingRequest *request,
//    FeatureRankingResponse *response) {
//    MatrixError err;
//    try {
//
//        switch (request->type()) {
//            case dg::REC_TYPE_VEHICLE:
//                return getRankedCarVector(request, response);
//            case dg::REC_TYPE_FACE:
//                return getRankedFaceVector(request, response);
//            case dg::REC_TYPE_ALL:
//                return getRankedAllVector(request, response);
//            case dg::REC_TYPE_DEFAULT:
//                return getRankedCarVector(request, response);
//
//            default:
//                LOG(ERROR) << "bad request(" << request->reqid() << "), unknown action";
//                err.set_code(-1);
//                err.set_message("bad request, unknown action");
//                return err;
//        }
//    }
//    catch (const std::exception &e) {
//        LOG(WARNING) << "bad request(" << request->reqid() << "), " << e.what() << endl;
//        err.set_code(-1);
//        err.set_message(e.what());
//        return err;
//    }
//}
//MatrixError RankerAppsService::getRankedAllVector(
//    const FeatureRankingRequest *request,
//    FeatureRankingResponse *response) {
//    vector<Score> faceScores;
//    vector<Score> carScores;
//    MatrixError err;
//    err = getFaceScoredVector(faceScores, request, response);
//    if (err.code() != 0) {
//        return err;
//    }
//
//    err = getCarScoredVector(carScores, request, response);
//    if (err.code() != 0) {
//        return err;
//    }
//
//    int limit = getLimit(request);
//    if (faceScores.size() > 0) {
//        partial_sort(faceScores.begin(), faceScores.begin() + limit,
//                     faceScores.end());
//        faceScores.resize(limit);
//    }
//    if (carScores.size() > 0) {
//
//        partial_sort(carScores.begin(), carScores.begin() + limit,
//                     carScores.end());
//        carScores.resize(limit);
//    }
//    for (Score &s : faceScores) {
//        response->add_ids(request->candidates(s.index_).id());
//        response->add_scores(s.score_);
//    }
//    for (Score &s : carScores) {
//        response->add_ids(request->candidates(s.index_).id());
//        response->add_scores(s.score_);
//    }
//}
//MatrixError RankerAppsService::getRankedCarVector(
//    const FeatureRankingRequest *request,
//    FeatureRankingResponse *response) {
//    vector<Score> scores;
//    MatrixError err;
//
//    err = getCarScoredVector(scores, request, response);
//
//    //sort & fill
//    sortAndFillResponse(request, scores, response);
//    return err;
//}
//MatrixError RankerAppsService::getCarScoredVector(
//    vector<Score> &scores, const FeatureRankingRequest *request,
//    FeatureRankingResponse *response) {
//    string prefix = requestPrefix(request);
//    response->set_reqid(request->reqid());
//
//    MatrixError err;
//
//    if (!request->has_image()) {
//        LOG(ERROR) << prefix << "image descriptor does not exist";
//        err.set_code(-1);
//        err.set_message("image descriptor does not exist");
//        return err;
//    }
//
//    Mat image;
//    err = ImageService::ParseImage(request->image(), image);
//    if (err.code() != 0) {
//        LOG(ERROR) << prefix << "parse image failed, " << err.message();
//        return err;
//    }
//
//    Rect hotspot = getHotspot(request, image);
//
//    vector<CarRankFeature> features;
//    err = extractFeatures(request, features, limits_);
//
//    if (err.code() != 0) {
//        LOG(ERROR) << prefix << "parse candidates failed, " << err.message();
//        return err;
//    }
//    vector<Rect> hotspots;
//    hotspots.push_back(hotspot);
//    CarRankFrame f(0, image, hotspots, features);
//
//    MatrixEnginesPool<SimpleRankEngine> *engine_pool = MatrixEnginesPool<SimpleRankEngine>::GetInstance();
//
//    EngineData data;
//    data.func = [&f, &data]() -> void {
//      return (bind(&SimpleRankEngine::RankCar, (SimpleRankEngine *) data.apps,
//                   placeholders::_1))(&f);
//    };
//
//    if (engine_pool == NULL) {
//        LOG(ERROR) << "Engine pool not initailized. " << endl;
//        return err;
//    }
//    engine_pool->enqueue(&data);
//    data.Wait();
//
//    scores = f.result_;
//
//    //scores = car_ranker_.Rank(image, hotspot, features);
//    return err;
//}


//void RankerAppsService::sortAndFillResponse(
//    const FeatureRankingRequest *request, vector<Score> &scores,
//    FeatureRankingResponse *response) {
//    if (scores.size() == 0) {
//        return;
//    }
//    int limit = getLimit(request);
//
//    partial_sort(scores.begin(), scores.begin() + limit, scores.end());
//    scores.resize(limit);
//    for (Score &s : scores) {
//        response->add_ids(request->candidates(s.index_).id());
//        response->add_scores(s.score_);
//    }
//}
//
//string RankerAppsService::requestPrefix(const FeatureRankingRequest *request) {
//    stringstream ss;
//    ss << "request(" << request->reqid() << "): ";
//    return ss.str();
//}
//
//Rect RankerAppsService::getHotspot(const FeatureRankingRequest *request,
//                                   const Mat &image) {
//    if (request->interestedareas_size() > 0) {
//        const Cutboard &cb = request->interestedareas(0);
//        if (cb.width() != 0 && cb.height() != 0) {
//            return Rect(cb.x(), cb.y(), cb.width(), cb.height());
//        }
//    }
//    return Rect(0, 0, image.cols, image.rows);
//}

//int RankerAppsService::getLimit(const FeatureRankingRequest *request) {
//    int limit = request->limit();
//    if (limit <= 0 || limit >= request->candidates_size()) {
//        limit = request->candidates_size();
//    }
//    return limit;
//}

}
