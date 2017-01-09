#include <alignment/align_dlib.h>
#include <alignment/align_cdnn.h>
#include <alignment/align_cdnn_caffe.h>
#include <stdexcept>
#include "dlib/image_processing.h"
#include "dlib/opencv.h"
#include "dlib/pixel.h"
#include "dlib_utils.h"
#include "caffe_interface.h"
#include "face_inf.h"
#include "post_filter.h"
#include "dgface_utils.h"
#include "dgface_config.h"

using namespace cv;
using namespace std;
namespace DGFace{
DlibAlignment::DlibAlignment(vector<int> face_size, const string &align_model, const std::string &det_type)
        : Alignment(face_size, false) {
    dlib::deserialize(align_model) >> _sp;
    _det_type = det_type;
}

DlibAlignment::~DlibAlignment(void) {
}


void DlibAlignment::align_impl(const Mat &img, const RotatedRect &rot_bbox, AlignResult &result) {
	align_impl(img, rot_bbox.boundingRect(), result);
}

void DlibAlignment::align_impl(const Mat &img, const Rect &bbox, AlignResult &result) {
    Rect reverse_box;
    if (_det_type == "ssd") {
        Rect adjust_box = bbox;
        const float h_rate = 0.4;
        reverse_box.height = adjust_box.height / (1-h_rate);
        reverse_box.y = adjust_box.y - reverse_box.height * h_rate;

        const float w_rate = 0.1;
        reverse_box.width = adjust_box.width / (1-w_rate*2);
        reverse_box.x = adjust_box.x - reverse_box.width * w_rate;
    } else if(_det_type == "rpn") {
        Rect adjust_box = bbox;
        const float h_rate = 0.3;
        reverse_box.height = adjust_box.height / (1-h_rate);
        reverse_box.y = adjust_box.y - reverse_box.height * h_rate;

        const float w_rate = 0.15;
        reverse_box.width = adjust_box.width / (1-w_rate*2);
        reverse_box.x = adjust_box.x - reverse_box.width * w_rate;


    }
    dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
    dlib::rectangle dlib_bbox;
    cv_rect2dlib(reverse_box, dlib_bbox);
    assert(dlib_bbox.right() <= img.cols);
    assert(dlib_bbox.bottom() <= img.rows);
    assert(dlib_bbox.left() > -1);
    assert(dlib_bbox.top() > -1);
    dlib::full_object_detection shape = _sp(dlib_img, dlib_bbox);
    result.landmarks.resize(0);
    if (shape.num_parts() < 1)
        return;
    auto &landmarks = result.landmarks;
    landmarks.reserve(shape.num_parts());
    for (size_t i = 0; i < shape.num_parts(); ++i) {
        dlib::point &p = shape.part(i);
        landmarks.emplace_back(p.x(), p.y());
    }
}

///////////////////////////////////////////////////////////////////////
CdnnAlignment::CdnnAlignment(vector<int> face_size, string modelDir):Alignment(face_size, false) {
	string real_model_dir = (modelDir.back() != '/') ? modelDir + "/" : modelDir;
    if(!Cdnn::InitFacialLandmarkModel(real_model_dir.c_str(), _keyPointsModel)) {
        cout << "Fail to init from " << real_model_dir << endl;
        exit(-1);
    }
}

CdnnAlignment::~CdnnAlignment(void) {
    Cdnn::ReleaseKeyPointsModel(_keyPointsModel);
}


void CdnnAlignment::landmarks_transform(const cv::Mat& trans_mat, 
										const std::vector<cv::Point2f>& src_landmarks, 
										std::vector<cv::Point2f>& dst_landmarks) {
	dst_landmarks.resize(src_landmarks.size());
	for(size_t pt_idx = 0; pt_idx < src_landmarks.size(); ++pt_idx) {
		Point2f& dst_pt = dst_landmarks[pt_idx];

		const double* trans_mat_pt = trans_mat.ptr<double>(0);
		dst_pt.x = trans_mat_pt[0] * src_landmarks[pt_idx].x + trans_mat_pt[1] * src_landmarks[pt_idx].y + trans_mat_pt[2];

		trans_mat_pt = trans_mat.ptr<double>(1);
		dst_pt.y = trans_mat_pt[0] * src_landmarks[pt_idx].x + trans_mat_pt[1] * src_landmarks[pt_idx].y + trans_mat_pt[2];
	} 
}

void CdnnAlignment::align_impl(const cv::Mat &img, 
                            const cv::RotatedRect& rot_bbox,
                            AlignResult &result) {

	Point2f face_center = rot_bbox.center;
	Size2f face_size = rot_bbox.size;
	float rot_angle = rot_bbox.angle;
	
	Mat rotate_mat = getRotationMatrix2D(face_center, rot_angle, 1);
	Mat rotated_img;
	// rotate the image 
	warpAffine(img, rotated_img, rotate_mat, img.size(), INTER_LINEAR,BORDER_CONSTANT,Scalar(128,128,128));

	Rect rectified_bbox = Rect(face_center.x - face_size.width * 0.5,
								face_center.y - face_size.height * 0.5,
								face_size.width,
								face_size.height);

	AlignResult rectified_result = {};
	align_impl(rotated_img, rectified_bbox, rectified_result);
	
	// rotate back the landmarks
	Mat inv_rotate_mat;
	invertAffineTransform(rotate_mat, inv_rotate_mat);
	landmarks_transform(inv_rotate_mat, rectified_result.landmarks, result.landmarks);

	// fill all landmarks' scores
	result.landmark_scores.resize(result.landmarks.size(), 1);

	result.score = rectified_result.score;
}

void CdnnAlignment::align_impl(const cv::Mat &img, 
                            const cv::Rect& bbox,
                            AlignResult &result) {

    CvRect face_crop(bbox);
    IplImage ipl_image = img;

    vector<double> landmarks;
    bool bRet = Cdnn::PredictFacialLandmarks(face_crop, landmarks, &ipl_image, _keyPointsModel, 8);
    if(!bRet) {
        cerr << "Fail to Alignment" << endl;
        landmarks.clear();
    }
    
    // calculate score
    Mat img4score = img.clone();
    vector<Point2f> pts4score;
    for(size_t i = 0; i < landmarks.size() / 2; ++i) {
        pts4score.push_back(Point2f(landmarks[i * 2], landmarks[i * 2 + 1]));
    }
    result.score = FaceScoreModel::scoring(img4score, pts4score);

    result.landmarks.resize(0);
	cvtLandmarks(landmarks, result.landmarks);

}

///////////////////////////////////////////////////////////////////////
CdnnCaffeAlignment::CdnnCaffeAlignment(vector<int> face_size, 
									string model_dir, 
									int gpu_id, 
									bool is_encrypt):Alignment(face_size, is_encrypt) {

    int argc = 1;
    char* argv[] = {""};
    _param.gpu      = gpu_id;

    vis::initcaffeglobal(argc, argv, gpu_id);

	string cfg_file;
	vector<string> deploy_files, model_files;
	addNameToPath(model_dir, "/align_cdnn_caffe.json", cfg_file);
	ParseConfigFile(cfg_file, deploy_files, model_files);

	for(int i = 0; i < 2; ++i) {
		addNameToPath(model_dir, "/" + deploy_files[i], deploy_files[i]);
		addNameToPath(model_dir, "/" + model_files[i], model_files[i]);
	}
	_cdnn_alignment = NULL;
    _cdnn_alignment = new FaceAlignWithConf(deploy_files, model_files);
	if(_cdnn_alignment == NULL) {
		cout << "can't initialize align_cdnn_caffe" << endl;
		exit(-1);
	}
}

void CdnnCaffeAlignment::ParseConfigFile(const string& cfg_file, 
										vector<string>& deploy_files,
										vector<string>& model_files) {
	deploy_files.clear();
	model_files.clear();

	string cfg_content;
	int ret = getConfigContent(cfg_file, false, cfg_content);
	if(ret != 0) {
		cout << "fail to decrypt config file." << endl;
		return;
	}

	stringstream ssin(cfg_content);
	for(int i = 0; i < 2; ++i) {
		string tmp_str;
		getline(ssin, tmp_str);
		deploy_files.push_back(tmp_str);

		getline(ssin, tmp_str);
		model_files.push_back(tmp_str);
	}

}


CdnnCaffeAlignment::~CdnnCaffeAlignment(void) {
    if (_cdnn_alignment) {
        delete _cdnn_alignment;
    }
}

void cvtRotatedRectToDetectInfo(const RotatedRect& rot_bbox, DetectedFaceInfo& det_info) {

    vector<float> rbox(5,0);
    float degree_rad = static_cast<float>(rot_bbox.angle) * CV_PI / 180;
    float cos_degree = cos(degree_rad),
          sin_degree = sin(degree_rad);

	det_info.width  = rot_bbox.size.width;
	det_info.height = rot_bbox.size.height;
	det_info.left   = rot_bbox.center.x - cos_degree*det_info.width/2 + sin_degree*det_info.height/2;
	det_info.top    = rot_bbox.center.y - sin_degree*det_info.width/2 - cos_degree*det_info.height/2;
	det_info.degree = rot_bbox.angle;

}

void CdnnCaffeAlignment::align_impl(const cv::Mat &img, 
									const cv::RotatedRect& rot_bbox,
									AlignResult &result) {

	DetectedFaceInfo detect_box = {};
	cvtRotatedRectToDetectInfo(rot_bbox, detect_box);

    // prepare image
	Mat safe_copy = img.clone();
    IplImage ipl_image = safe_copy;
    
    // clear result
    result.landmarks.clear();
    LandMarkInfo landmark_info = {};

    // align
    int ret = _cdnn_alignment->Align(&ipl_image, _param, detect_box, landmark_info);

    if (ret) {
        cerr << "Fail to Alignment" << endl;
        result.landmarks.clear();
        return;
    }

	// landmark  
	cvtLandmarks(landmark_info.landmarks, result.landmarks);	
	result.landmark_scores = landmark_info.landmark_scores;
	
    result.score = landmark_info.score;
}


void CdnnCaffeAlignment::align_impl(const cv::Mat &img, 
                            const cv::Rect& bbox,
                            AlignResult &result) {
    
    // prepare detection box
    DetectedFaceInfo detect_box = {};
    detect_box.left     = bbox.x;
    detect_box.top      = bbox.y;
    detect_box.width    = bbox.width;
    detect_box.height   = bbox.height;
	detect_box.degree	= 0;

    // prepare image
    IplImage ipl_image = img;
    
    // clear result
    result.landmarks.clear();
    LandMarkInfo landmark_info = {};

    // align
    int ret = _cdnn_alignment->Align(&ipl_image, _param, detect_box, landmark_info);
    if (ret) {
        cerr << "Fail to Alignment" << endl;
        result.landmarks.clear();
        return;
    } 

	cvtLandmarks(landmark_info.landmarks, result.landmarks);	
	result.landmark_scores = landmark_info.landmark_scores;
   
    result.score = landmark_info.score;
}

///////////////////////////////////////////////////////////////
/*--------------------
Alignment *create_alignment(const string &prefix) {
    Config *config = Config::instance();
    string type    = config->GetConfig<string>(prefix + "alignment", "dlib");
    vector<int> face_size = config->GetConfigArr(prefix + "alignment.face_size",
            vector<int> {128});
    if (type == "dlib") {
		throw new runtime_error("don't use dlib");
        string model = config->GetConfig<string>(prefix + "alignment.dlib.model");
        string det_type = config->GetConfig<string>(prefix + "detector");
        return new DlibAlignment(face_size, model, det_type);
    } else if (type == "cdnn") {
        string model_dir = config->GetConfig<string>(prefix + "alignment.cdnn.model_dir");
        return new CdnnAlignment(face_size, model_dir);
    } else if (type == "cdnn_caffe") {
        string model_dir = config->GetConfig<string>(prefix + "alignment.cdnn_caffe.model_dir");
        int gpu_id         = config->GetConfig(prefix + "alignment.cdnn_caffe.gpu_id", 0);// -1 for CPU, 0~3 for GPU
        return new CdnnCaffeAlignment(face_size, model_dir, gpu_id);
    }
    throw new runtime_error("unknown alignment");
}
*/

Alignment *create_alignment_with_global_dir(const align_method& method, 
										const string& global_dir,
										int gpu_id, 
										bool is_encrypt, 
										int batch_size) {
const std::map<align_method, std::string> align_map {
	{align_method::DLIB, "DLIB"},
	{align_method::CDNN, "CDNN"},
	{align_method::CDNN_CAFFE, "CDNN_CAFFE"}
};
	string align_key = "FaceAlignment";
	string full_key = align_key + "/" + align_map.at(method);

	string config_file;
	addNameToPath(global_dir, "/"+getGlobalConfig(), config_file); 
	Config config;
	config.Load(config_file);
	string local_model_path = static_cast<string>(config.Value(full_key));
	if(local_model_path.empty()){
		throw new runtime_error(full_key + " not exist!");
	} else {
		string tmp_model_dir = is_encrypt ? getEncryptModelDir() : getNonEncryptModelDir() ;	
		string model_path; 
		addNameToPath(global_dir, "/"+tmp_model_dir+"/"+local_model_path, model_path); 
		return create_alignment(method, model_path, gpu_id, is_encrypt, batch_size);
	}
}

Alignment *create_alignment_with_config(const align_method& method, 
									const string& config_file,
									int gpu_id, 
									bool is_encrypt, 
									int batch_size) {
const std::map<align_method, std::string> align_map {
	{align_method::DLIB, "DLIB"},
	{align_method::CDNN, "CDNN"},
	{align_method::CDNN_CAFFE, "CDNN_CAFFE"}
};
	string align_key = "FaceAlignment";
	string full_key = align_key + "/" + align_map.at(method);

	Config path_cfg;
	path_cfg.Load(config_file);
	string model_path = static_cast<string>(path_cfg.Value(full_key));
	if(model_path.empty()){
		throw new runtime_error(full_key + " not exist!");
	} else {
		return create_alignment(method, model_path, gpu_id, is_encrypt, batch_size);
	}
}
Alignment *create_alignment(const align_method& method, 
							const string& model_dir,
							int gpu_id, 
							bool is_encrypt, 
							int batch_size) {
	vector<int> face_size = {128};
	
	switch(method) {
		case align_method::CDNN: {
			face_size[0] = 600;
			return new CdnnAlignment(face_size, model_dir);
			break;
		}
		case align_method::CDNN_CAFFE: {
			face_size[0] = 600;
			return new CdnnCaffeAlignment(face_size, model_dir, gpu_id, is_encrypt);
			break;
		}
		case align_method::DLIB: {
			throw new runtime_error("don't use dlib");
			break;
		}
		default:
			throw new runtime_error("unknown alignment");
	}
	// if(method == "cdnn") {
	// 	face_size[0] = 600;
    //     return new CdnnAlignment(face_size, model_dir);
	// } else if(method == "cdnn_caffe") {
	// 	face_size[0] = 600;
    //     return new CdnnCaffeAlignment(face_size, model_dir, gpu_id);
	// } else if(method == "dlib") {
	// 	throw new runtime_error("don't use dlib");
	// }
    // throw new runtime_error("unknown alignment");
}
}
