#include <recognition.h>
#include <verification.h>
#include <tracking.h>
#include "dlib/optimization/max_cost_assignment.h"

using namespace std;
using namespace cv;

namespace DGFace{
const float ALPHA = 0.9;
const int KEEP_FEATURE_NUM = 20; //3;
const int MAX_LIFE_TIME    = 5; //10;

Tracking::Tracking(Recognition *recog, Verification *verify, float thresh)
        : _next_id(0), _frame_index(0), _recognize(recog),
        _verifier(verify), _verify_thresh(thresh) {
    assert(thresh >= 0);
}

Tracking::~Tracking() {
    delete _recognize;
    delete _verifier;
}

void Tracking::update(const Mat &img) {
    _bbox_cache.clear();
    _feature_cache.clear();
    _match_cache.clear();
    _similariy_cache.clear();
    _frame_index++;
    find_objects(img, _bbox_cache);
    //if (_bbox_cache.size() < 1) {
    //    _objects.clear();
    //    return;
    //}
    // for (auto &bbox : _bbox_cache) {
    //     assert(bbox.height + bbox.y <= img.rows);
    //     assert(bbox.width + bbox.x <= img.cols);
    //     assert(bbox.x >= 0);
    //     assert(bbox.y >= 0);
    // }
    extract_features(img, _bbox_cache, _feature_cache);
    if (_objects.size() != 0) {
        build_similarity(_bbox_cache, _feature_cache, _similariy_cache);
        match_object(_similariy_cache, _match_cache);
    }
    post_process(_match_cache, _bbox_cache, _feature_cache);
    _bbox_cache.clear();
    _feature_cache.clear();
    _match_cache.clear();
    _similariy_cache.clear();
}

void Tracking::extract_features(const Mat &img, const vector<Rect> &bboxes,
        vector<FeatureType> &features) {
    vector<RecogResult> recog_result;
    features.clear();
    features.reserve(bboxes.size());
    _image_cache.clear();
    _image_cache.reserve(bboxes.size());
    for (size_t i = 0; i < bboxes.size(); i++) {
        auto &bbox = bboxes[i];
        _image_cache.push_back(img(bbox));
    }
    _recognize->recog(_image_cache, recog_result, "NONE");
    for (size_t i = 0; i < recog_result.size(); i++) {
        features.push_back(recog_result[i].face_feat);
    }
    _image_cache.clear();
}

// TODO: fix the pipeline
void Tracking::extract_features(const Mat &img, const vector<RotatedRect> &rot_bboxes,
        vector<FeatureType> &features) {
    vector<RecogResult> recog_result;
    features.clear();
    features.reserve(rot_bboxes.size());
    _image_cache.clear();
    _image_cache.reserve(rot_bboxes.size());
    for (size_t i = 0; i < rot_bboxes.size(); i++) {
        auto &bbox = rot_bboxes[i];
        _image_cache.push_back(img(bbox.boundingRect()));
    }
    _recognize->recog(_image_cache, recog_result, "NONE");
    for (size_t i = 0; i < recog_result.size(); i++) {
        features.push_back(recog_result[i].face_feat);
    }
    _image_cache.clear();
}

// TODO: just a warpper
void Tracking::build_similarity(const vector<RotatedRect> &now_rot_bboxes,
        const vector<FeatureType> &now_features,
        vector<vector<float> > &similarity) {
	vector<Rect> warp_bboxes(now_rot_bboxes.size());
	for(size_t i = 0; i < now_rot_bboxes.size(); ++i) {
		warp_bboxes[i] = now_rot_bboxes[i].boundingRect();
	}
	build_similarity(warp_bboxes, now_features, similarity);
}

void Tracking::build_similarity(const vector<Rect> &now_bboxes,
        const vector<FeatureType> &now_features,
        vector<vector<float> > &similarity) {
    size_t last_objs = _objects.size();
    size_t now_objs  = now_bboxes.size();
    assert(now_bboxes.size() == now_features.size());
    similarity.resize(last_objs);
    // cout << _frame_index << endl;
    for (size_t obj_idx = 0; obj_idx < last_objs; obj_idx++) {
        const auto &object = _objects[obj_idx];
        const Rect &obj_bbox = object.bbox;
        similarity[obj_idx].resize(now_objs);
        for (size_t now_idx = 0; now_idx < now_objs; now_idx++) {
            float max_simi = 0;
            auto &now_feat = now_features[now_idx];
            for (auto &last_feat : object.last_features) {
                float simi = _verifier->verify(last_feat, now_feat);
                max_simi = max(max_simi, simi);
            }
            if (max_simi >= _verify_thresh) {
                const Rect &now_bbox = now_bboxes[now_idx];
                cv::Rect intersect = obj_bbox & now_bbox;
                float iou = intersect.area() * 1.0f /
                    (obj_bbox.area() + now_bbox.area() - intersect.area());
                similarity[obj_idx][now_idx] =
                    ALPHA * max_simi + (1 - ALPHA) * iou;
                //cout << "max_simi: " << max_simi << ", iou: " << iou << ", score: " << similarity[obj_idx][now_idx] << endl;
                //cout << "=============================" << endl;
            } else {
                similarity[obj_idx][now_idx] = 0;
            }
            // cout << similarity[obj_idx][now_idx] << " ";
        }
        // cout << endl;
    }
}



// TODO: just a warpper
void Tracking::post_process(const vector<ssize_t> &match_result,
        const vector<RotatedRect> &now_rot_bboxes,
        const vector<FeatureType> &now_features) {
	vector<Rect> warp_bboxes(now_rot_bboxes.size());
	for(size_t i = 0; i < now_rot_bboxes.size(); ++i) {
		warp_bboxes[i] = now_rot_bboxes[i].boundingRect();
	}
	post_process(match_result, warp_bboxes, now_features);
}

void Tracking::post_process(const vector<ssize_t> &match_result,
        const vector<Rect> &now_bboxes,
        const vector<FeatureType> &now_features) {
    size_t last_objs = _objects.size();
    size_t now_objs  = now_bboxes.size();
    assert(match_result.size() == _objects.size());
    assert(now_bboxes.size() == now_features.size());
    _matched_flags.clear();
    _matched_flags.resize(now_objs, false);
    for (size_t idx = 0; idx < last_objs; idx++) {
        ssize_t matched = match_result[idx];
        if (matched < 0)
            continue;
        auto &object = _objects[idx];
        _matched_flags[matched] = true;
        object.last_seen = _frame_index;
        object.bbox = now_bboxes[matched];
        object.last_features.push_back(now_features[matched]);
        while (object.last_features.size() > KEEP_FEATURE_NUM) {
            object.last_features.pop_front();
        }
    }

    for (size_t idx = 0; idx < now_objs; idx++) {
        if (_matched_flags[idx])
            continue;
        TrackedObj object;
        object.last_seen = _frame_index;
        object.obj_id    = _next_id++;
        object.bbox      = now_bboxes[idx];
        object.last_features.push_back(now_features[idx]);
        _objects.push_back(object);
    }

    size_t index = 0;
    //cout << "obj size: " << _objects.size() << " frame index: " << _frame_index << endl;
    for (size_t i = 0; i < _objects.size(); i++) {
        //cout << "last_seen: " <<  _objects[i].last_seen << endl;
        if (_frame_index - _objects[i].last_seen <= MAX_LIFE_TIME) {
            if (i != index) {
                _objects[index] = std::move(_objects[i]);
            }
            index++;
        }
    }
    _objects.resize(index);
    // cout << _frame_index << "\t" << now_objs << "\t" <<  _objects.size() << endl;
}

void Tracking::match_object(const vector<vector<float> >&similarity,
        vector<ssize_t> &match_result) {
    size_t row = similarity.size();
    size_t col = similarity[0].size();
    size_t n   = max(row, col);
    dlib::matrix<size_t> cost(n, n);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = 0; c < n; c++) {
            if (r >= row or c >= col) {
                cost(r, c) = 0;
            } else {
                cost(r, c) = similarity[r][c] * 10000000;
            }
        }
    }
    std::vector<long> result;
    result = max_cost_assignment(cost);
    assert(result.size() == n);
    result.resize(row);
    match_result.resize(row);
    for (size_t idx = 0; idx < row; idx++) {
        size_t matched = result[idx];
        if (matched >= col || cost(idx, matched) < 1e-5) {
            match_result[idx] = -1;
        } else {
            match_result[idx] = result[idx];
        }
    }
}
}
