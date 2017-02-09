#include "dgface_utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <utility>

using namespace std;
using namespace cv;

namespace DGFace {

//////////////////////////////////////////////////////////////////
// decrypt function
//////////////////////////////////////////////////////////////////
int getConfigContent(string file, bool is_encrypt, string& content) {
	FILE *fp = fopen(file.c_str(), "rb");
	if(fp == NULL) {
		return -1;
	}

	long length;
	fseek(fp, 0L, SEEK_END);
	length = ftell(fp);
	fseek(fp, 0L, SEEK_SET);

	unsigned char *src = (unsigned char *)calloc(length, 1);
	length = fread(src, sizeof(char), length, fp);
	
	if (is_encrypt){
		unsigned char * dst = (unsigned char *)calloc(length, 1);
		DecryptModel((unsigned char *)src, length, dst);
		// content((const char *)dst);
		content = string((char *)dst);
		free(dst);
	} else {
		// content((const char *) src);
		content = string((char *) src );
		//content.assign(ch_src, (int)length);
	}
	free(src);
	fclose(fp);
	return 0;
}

long FileSize(const string &file) {
	ifstream is;
	is.open(file.c_str(), ios::binary|ios::ate);
	
	// get length of file:
	is.seekg(0, std::ios::end);
	
	long length = is.tellg();
	is.seekg(0, std::ios::beg);
	is.close();
	return length;
}

string ReadStringFromFile(const string& filePath, const string& mode) {
	long length = FileSize(filePath);
	FILE *file = fopen(filePath.c_str(), mode.c_str());
	if (file == NULL) {
		return "";
	}
	char *data = (char *)malloc(length);
	memset(data, 0, length);
	length = fread(data, sizeof(char), length, file);
	fclose(file);
	string result(data,length);
	free(data);
	return result;
}

// Actually, getFileContent == getConfigContent 
int getFileContent(string file, bool is_encrypt, string& content) {
	// return getConfigContent(file, is_encrypt, content);
	if (!is_encrypt) {
		content = ReadStringFromFile(file, "rb");
        if(content == "")
            return -1;
        else
            return 0;
	} else {
		int length;
		length = FileSize(file);
		FILE *fp = fopen(file.c_str(), "rb");
		if (fp == NULL) {
		return -1;
		}
		unsigned char *src = (unsigned char *)calloc(length, 1);
		
		length = fread(src, sizeof(char), length, fp);
		fclose(fp);
		unsigned char * dst = (unsigned char *)calloc(length, 1);
		DecryptModel((unsigned char *)src, length, dst);
		if (file.back() == 't') {
			string content_tmp((char *)dst);
			content = content_tmp;
		} else {
			string content_tmp((char *)dst, length);
			content = content_tmp;
		}
		
		free(src);
		free(dst);
	}
}

void addNameToPath(const string& model_dir, const string& name, string& appended_dir) {
	appended_dir.clear();
	if(model_dir.empty()) {
		return;
	}
	if(model_dir.back() == '$') {
		size_t pos = model_dir.length() - 1;
		appended_dir = model_dir;

		size_t sublen = (name.back() == '$') ? name.length()-1 : name.length();
		appended_dir.insert(pos, name, 0, sublen);
	} else {
		appended_dir = model_dir + name;
	}
}
//////////////////////////////////////////////////////////////////
// draw function
//////////////////////////////////////////////////////////////////
void drawRotatedRect(cv::Mat& draw_img, const cv::RotatedRect& rot_bbox) {
	Point2f vertices[4];
	rot_bbox.points(vertices);
	for (int i = 0; i < 4; i++) {
		line(draw_img, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));
	}
}

void drawDetectionResult(cv::Mat& draw_img, const DetectResult& det_result, bool display_score) {
	const vector<RotatedBbox>& rot_bbox = det_result.boundingBox;
	for(size_t i = 0; i < rot_bbox.size(); ++i) {
		RotatedRect rot_face = rot_bbox[i].second;
		drawRotatedRect(draw_img, rot_face);
		if(display_score) {
			Point org = rot_face.boundingRect().tl();
			char conf_str[16];
			sprintf(conf_str, "%.4f", rot_bbox[i].first);
			putText(draw_img,string(conf_str), org, FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0));
		}
	}
}

void drawLandmarks(Mat& draw_img, const AlignResult &align_result) {
	auto &landmarks = align_result.landmarks;
	for (auto pt = landmarks.begin(); pt != landmarks.end(); ++pt) {
		circle(draw_img, *pt, 2, Scalar(0,255,0), -1);
	}
}

//////////////////////////////////////////////////////////////////
// save and load function
//////////////////////////////////////////////////////////////////
bool saveFeature(const string& fea_log, const vector<RecogResult>& recog_results) {
	ofstream feafile(fea_log);
	if(!feafile) {
		return false;
	}
	
	for(size_t i = 0; i < recog_results.size(); ++i) {
		for(size_t j = 0; j < recog_results[i].face_feat.size(); ++j) {
				feafile << recog_results[i].face_feat[j] << " " ;
		}
		feafile << endl;
	}
	feafile.close();
	return true;
}

bool dimConsist(const vector<RecogResult>& recog_results) {
	int fea_dim = -1;
	for(size_t i = 0; i < recog_results.size(); ++i) {
		const RecogResult& curr_recog = recog_results[i];
		if(curr_recog.face_feat.empty()) {
			continue; // skip empty feature
		} else if(fea_dim == -1){
			fea_dim = static_cast<int>(curr_recog.face_feat.size());
		} else if(fea_dim != static_cast<int>(curr_recog.face_feat.size())){
			cout << "feature dimension not match" << endl;
			return false;	
		}
	}
	return true;
}
bool loadFeature(const string& fea_log, vector<RecogResult>& recog_results) {
	ifstream feafile(fea_log);
	if(!feafile) {
		cout << "can't open " << fea_log << endl;
		return false;
	}

	recog_results.clear();
	string line;
	while(getline(feafile, line)) {
		RecogResult one_recog_result;
		if(line.empty()) {
			one_recog_result.face_feat.clear();
		} else {
			stringstream feat_ss(line);
			FeatureElemType feat_data;
			while(feat_ss >> feat_data) {
				one_recog_result.face_feat.push_back(feat_data);
			}
		}
		recog_results.push_back(one_recog_result);
	}
	
	feafile.close();
	return dimConsist(recog_results);
}

//////////////////////////////////////////////////////////////////
// convert function
//////////////////////////////////////////////////////////////////
void cvtPoint2iToPoint2f(const std::vector<cv::Point>& pts_2i, std::vector<cv::Point2f>& pts_2f) {
	pts_2f.resize(pts_2i.size());
	for(size_t i = 0; i < pts_2i.size(); ++i) {
		pts_2f[i].x = static_cast<float>(pts_2i[i].x);
		pts_2f[i].y = static_cast<float>(pts_2i[i].y);
	}
}

void cvtPoint2fToPoint2i(const std::vector<cv::Point2f>& pts_2f, std::vector<cv::Point>& pts_2i) {
	pts_2i.resize(pts_2f.size());
	for(size_t i = 0; i < pts_2f.size(); ++i) {
		pts_2i[i].x = static_cast<int>(pts_2f[i].x);
		pts_2i[i].y = static_cast<int>(pts_2f[i].y);
	}
}


void cvtLandmarks(const std::vector<cv::Point2f>& src_landmarks, std::vector<Point_2d_f>& dst_landmarks) {
	dst_landmarks.clear();
	dst_landmarks.reserve(src_landmarks.size());
	for(size_t i = 0; i < src_landmarks.size(); ++i) {
		Point_2d_f pt = {src_landmarks[i].x, src_landmarks[i].y};
		dst_landmarks.push_back(move(pt));
	}
}

void cvtLandmarks(const std::vector<Point_2d_f>& src_landmarks, std::vector<cv::Point2f>& dst_landmarks) {
	dst_landmarks.clear();
	dst_landmarks.reserve(src_landmarks.size());
	for(size_t i = 0; i < src_landmarks.size(); ++i) {
		Point2f pt(src_landmarks[i].x, src_landmarks[i].y);
		dst_landmarks.push_back(move(pt));
	}
}

void cvtLandmarks(const std::vector<cv::Point2f>& src_landmarks, std::vector<double>& dst_landmarks) {
	dst_landmarks.clear();
	for(size_t i = 0; i < src_landmarks.size(); ++i) {
		dst_landmarks.push_back(src_landmarks[i].x);
		dst_landmarks.push_back(src_landmarks[i].y);
	}
}

void cvtLandmarks(const std::vector<double>& src_landmarks, std::vector<cv::Point2f>& dst_landmarks) {
	dst_landmarks.clear();
	if(src_landmarks.size() % 2 != 0) {
		return;
	}
	dst_landmarks.reserve(src_landmarks.size() / 2);
	for(size_t i = 0; i < src_landmarks.size() / 2; ++i) {
		Point2f pt;
		pt.x = static_cast<float>(src_landmarks[i * 2]);
		pt.y = static_cast<float>(src_landmarks[i * 2 + 1]);
		dst_landmarks.push_back(move(pt));
	}
}

void cvtLandmarks(const std::vector<Point_2d_f>& src_landmarks, std::vector<double>& dst_landmarks) {
	dst_landmarks.clear();
	for(size_t i = 0; i < src_landmarks.size(); ++i) {
		dst_landmarks.push_back(src_landmarks[i].x);
		dst_landmarks.push_back(src_landmarks[i].y);
	}
}

void cvtLandmarks(const std::vector<double>& src_landmarks, std::vector<Point_2d_f>& dst_landmarks) {
	dst_landmarks.clear();
	if(src_landmarks.size() % 2 != 0) {
		return;
	}
	dst_landmarks.reserve(src_landmarks.size() / 2);
	for(size_t i = 0; i < src_landmarks.size() / 2; ++i) {
		Point_2d_f pt;
		pt.x = static_cast<float>(src_landmarks[i * 2]);
		pt.y = static_cast<float>(src_landmarks[i * 2 + 1]);
		dst_landmarks.push_back(move(pt));
	}
}

//////////////////////////////////////////////////////////////////
// evaluation function
//////////////////////////////////////////////////////////////////
float computeAP(const vector<float>& scores, const vector<bool>& trues) {
	if(scores.size() != trues.size()) {
		cout << "score size and label size not match." << endl;
		return -2;
	}

	vector<pair<float, bool> > score_label_vec(scores.size());
	for(size_t i = 0; i < score_label_vec.size(); ++i) {
		score_label_vec[i].first  = scores[i];
		score_label_vec[i].second = trues[i];
	}
	std::sort(score_label_vec.begin(), score_label_vec.end(), [](const pair<float, bool> &left, const pair<float, bool> &right) {
	return left.first > right.first;
	});
	
	// find out the truth
	vector<int> label_index(0);
	for(size_t i = 0; i < score_label_vec.size(); ++i) {
		if(score_label_vec[i].second == true) {
			label_index.push_back(i);
		}
	}

	// compute the AP
	float ap = 0.0f;
	for(size_t i = 0; i < label_index.size(); ++i) {
		ap += static_cast<float>(i + 1) / (label_index[i] + 1);
	}
	if(label_index.size() > 0) {
		ap /= label_index.size();
		return ap;
	} else {
		return -1;
	}
}


float computeMAP(const vector<vector<float> >& score_vec, const vector<vector<bool> >& true_vec) {
	vector<float> AP_vec;
	return computeMAP(score_vec, true_vec, AP_vec);
}
float computeMAP(const vector<vector<float> >& score_vec, const vector<vector<bool> >& true_vec, vector<float>& AP_vec) {
	AP_vec.resize(score_vec.size());
	if(score_vec.size() != true_vec.size()) {
		return -1;
	}

	for(size_t i = 0; i < score_vec.size(); ++i) {
		const vector<float>& curr_scores = score_vec[i];
		const vector<bool>& curr_trues = true_vec[i];
		float curr_ap = 0.0f;
		if(curr_scores.size() > 0){
			curr_ap = computeAP(curr_scores, curr_trues);
		} // curr_scores.size() == 0 means the not feature extracted from a query
		if(curr_ap == -2) {
			return -1;
		}
		AP_vec[i] = curr_ap;
	}
	
	float map = 0;
	int cnt = 0;
	for(auto& ap : AP_vec) {
		if(ap >= 0) {
			map += ap;
			++cnt;
		}
	}
	map /= cnt;
	if(cnt > 0) {
		return map;	
	} else {
		cout << "AP list is empty." << endl;
		return -1;
	}
}

// compute top1 ROC
void computeROC(const vector<vector<float> >& score_vec, const vector<vector<bool> >& true_vec, vector<pair<float, float> >& ROC_points, const float step) {
    assert(score_vec.size() == true_vec.size());
    vector<float> scores;
    vector<bool> trues;
    scores.clear();
    trues.clear();
    for (size_t id = 0; id < score_vec.size(); id++) {
        vector<pair<float, bool> > score_label_vec(score_vec[id].size());
        for(size_t i = 0; i < score_label_vec.size(); ++i) {
            score_label_vec[i].first  = score_vec[id][i];
            score_label_vec[i].second = true_vec[id][i];
        }
        std::sort(score_label_vec.begin(), score_label_vec.end(), [](const pair<float, bool> &left, const pair<float, bool> &right) {
            return left.first > right.first;
        });
        scores.push_back(score_label_vec[0].first);
        trues.push_back(score_label_vec[0].second);
    }

	return computeROC(scores, trues, ROC_points, step);
}

void computeROC(const vector<float>& scores, const vector<bool>& trues, vector<pair<float, float> >& ROC_points, const float step) {
    size_t num_data = scores.size();
    int num_P = 0;
    int num_N = 0;
    for (size_t i = 0; i < num_data; i++) {
        if (trues[i]) num_P++;
        else num_N++;
    } 

    for (float thr = 0.f; thr < 1.0f; thr += step) {
        int TP = 0;
        int FP = 0; 
        for (size_t i = 0; i < num_data; i++) {
            if (scores[i] > thr && trues[i]) TP++;
            else if (scores[i] > thr && (!trues[i])) FP++;  
        }
        //cout << "thr: " << thr << " TP: " << TP << " P: " << num_P << " FP: " << FP << " N: " << num_N << " TPR: " << TP/(float(num_P) + 1e-5) << "NPR: " << FP/(float(num_N) + 1e-5) << endl; 
        ROC_points.push_back(make_pair(FP/(float(num_N) + 1e-5), TP/(float(num_P) + 1e-5)));
    }
}
}
