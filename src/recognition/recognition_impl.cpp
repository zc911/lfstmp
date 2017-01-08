#include <recognition/recog_cnn.h>
#include <recognition/recog_lbp.h>
#include <recognition/recog_cdnn.h>
#include <recognition/recog_cdnn_caffe.h>
#include <recognition/caffe_batch_wrapper.h>
#include <recognition/recog_fuse.h>
#include <stdexcept>
#include "caffe_interface.h"

#include <transformation/trans_cdnn.h>
#include <transformation/trans_cdnn_caffe.h>
#include "dgface_utils.h"
#include "dgface_config.h"

using namespace std;
using namespace cv;
using namespace caffe;
namespace DGFace {

/*======================= CNN recognition  ========================= */
CNNRecog::CNNRecog(const string& model_file,
                   const string& trained_file,
                   const string& layer_name,
                   const vector<float> mean,
                   const float pixel_scale,
                   const bool use_GPU,
                   const int  gpu_id)
    : Recognition(false), _net(nullptr), _useGPU(use_GPU), _pixel_means(mean),
      _pixel_scale(pixel_scale), _layer_name(layer_name) {
    if (use_GPU) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    /* Load the network. */
    cout << "loading " << model_file << endl;
    _net.reset(new Net<float>(model_file, TEST));
    cout << "loading " << trained_file << endl;
    _net->CopyTrainedLayersFrom(trained_file);

    Blob<float>* input_blob = _net->input_blobs()[0];
    _batch_size = input_blob->num();

    _num_channels  = input_blob->channels();
    CHECK(_num_channels == 3 || _num_channels == 1)
            << "Input layer should have 1 or 3 channels.";
}

CNNRecog::~CNNRecog(void) {
}
void CNNRecog::recog_impl(const std::vector<cv::Mat>& faces,
                          const std::vector<AlignResult>& alignment,
                          std::vector<RecogResult>& results) {
    recog_impl(faces, results);
}
void CNNRecog::recog_impl(const vector<Mat> &faces, vector<RecogResult> &results) {
    Blob<float>* input_blob = _net->input_blobs()[0];
    //assert((int)faces.size() <= _batch_size);
    vector<int> shape = input_blob->shape();
    if (shape[0] != _batch_size) {
        shape[0] = static_cast<int>(faces.size());
        input_blob->Reshape(shape);
        _net->Reshape();
    }

    results.resize(faces.size());
    float* input_data = input_blob->mutable_cpu_data();
    for (size_t i = 0; i < faces.size(); i++)
    {
        Mat sample;
        Mat face = faces[i];
        //assert(face.cols == input_blob->width() && face.rows == input_blob->height());
        if (face.cols != input_blob->width() || face.rows != input_blob->height())
            resize(face, face, Size(input_blob->width(), input_blob->height()));

        if (face.channels() == 3 && _num_channels == 1)
            cvtColor(face, sample, CV_BGR2GRAY);
        else if (face.channels() == 4 && _num_channels == 1)
            cvtColor(face, sample, CV_BGRA2GRAY);
        else if (face.channels() == 4 && _num_channels == 3)
            cvtColor(face, sample, CV_BGRA2BGR);
        else if (face.channels() == 1 && _num_channels == 3)
            cvtColor(face, sample, CV_GRAY2BGR);
        else
            sample = face;

        size_t image_off = i * sample.channels() * sample.rows * sample.cols;
        for (int k = 0; k < sample.channels(); k++)
        {
            size_t channel_off = k * sample.rows * sample.cols;
            for (int row = 0; row < sample.rows; row++)
            {
                size_t row_off = row * sample.cols;
                for (int col = 0; col < sample.cols; col++)
                {
                    input_data[image_off + channel_off + row_off + col] =
                        (float(sample.at<uchar>(row, col * sample.channels() + k)) - _pixel_means[k]) / _pixel_scale;
                }
            }
        }
    }

    _net->ForwardPrefilled();
    if (_useGPU)
    {
        cudaDeviceSynchronize();
    }

    auto output_blob = _net->blob_by_name(_layer_name);
    const float *output_data = output_blob->cpu_data();
    const int feature_len = output_blob->channels();
    assert(feature_len > 1);
    // cerr << "output_blob:\t" << output_blob->count() << endl;
    results.resize(faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        const float *data = output_data + i * feature_len;
        FeatureType &feature = results[i].face_feat;
        feature.resize(feature_len);
        for (int idx = 0; idx < feature_len; ++idx) {
            feature[idx] = data[idx];
        }
    }
}

/*======================= LBP recognition  ========================= */
static Mat
histc(const Mat& src, int minVal = 0, int maxVal = 255, bool normed = false)
{
    Mat result;
    // Establish the number of bins.
    int histSize = maxVal - minVal + 1;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
    const float* histRange = { range };
    // calc histogram
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
    // normalize
    if (normed) {
        result /= (int)src.total();
    }
    return result.reshape(1, 1);
}

static Mat spatial_histogram(const Mat &src, int numPatterns,
                             int grid_x, int grid_y, bool /*normed*/)
{
    // calculate LBP patch size
    int width = src.cols / grid_x;
    int height = src.rows / grid_y;
    // allocate memory for the spatial histogram
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // return matrix with zeros if no data was given
    assert (!src.empty());
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    for (int i = 0; i < grid_y; i++) {
        for (int j = 0; j < grid_x; j++) {
            Mat src_cell = Mat(src, Range(i * height, (i + 1) * height), Range(j * width, (j + 1) * width));
            Mat cell_hist = histc(src_cell, 0, (numPatterns - 1), true);
            // copy to the result matrix
            Mat result_row = result.row(resultRowIdx);
            cell_hist.reshape(1, 1).convertTo(result_row, CV_32FC1);
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return result.reshape(1, 1);
}

static void elbp(const Mat &src, Mat &dst, int radius, int neighbors) {
    // allocate memory for result
    dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
    // zero
    dst.setTo(0);
    for (int n = 0; n < neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for (int i = radius; i < src.rows - radius; i++) {
            for (int j = radius; j < src.cols - radius; j++) {
                // calculate interpolated value
                float t = static_cast<float>(
                              w1 * src.at<uchar>(i + fy, j + fx) +
                              w2 * src.at<uchar>(i + fy, j + cx) +
                              w3 * src.at<uchar>(i + cy, j + fx) +
                              w4 * src.at<uchar>(i + cy, j + cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

LbpRecog::LbpRecog(int radius, int neighbors, int grid_x, int grid_y)
    : Recognition(false), _radius(radius), _neighbors(neighbors), _grid_x(grid_x), _grid_y(grid_y) {
}
void LbpRecog::recog_impl(const std::vector<cv::Mat>& faces,
                          const std::vector<AlignResult>& alignment,
                          std::vector<RecogResult>& results) {
    recog_impl(faces, results);
}
void LbpRecog::recog_impl(const vector<Mat> &faces, vector<RecogResult> &results) {
    results.clear();
    results.resize(faces.size());
    for (size_t i = 0; i < faces.size(); i++)
    {
        Mat sample;
        Mat face = faces[i];
        if (face.channels() == 3)
            cvtColor(face, sample, CV_BGR2GRAY);
        else if (face.channels() == 4)
            cvtColor(face, sample, CV_BGRA2GRAY);
        else
            sample = face;

        Mat lbp_img;
        elbp(sample, lbp_img, _radius, _neighbors);
        Mat feature = spatial_histogram(
                          Mat_<float>(lbp_img), /* lbp_image */
                          static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))),
                          /* number of possible patterns */
                          _grid_x, /* grid size x */
                          _grid_y, /* grid size y */
                          true /* normed histograms */);
        feature = feature.reshape(1, 1);
        auto &result = results[i].face_feat;
        assert(feature.rows == 1);
        assert(feature.channels() == 1);
        result.resize(feature.cols);
        for (int i = 0; i < feature.cols; i++) {
            result[i] = feature.at<float>(i);
        }
    }
}

/*static vector<float> read_mean(Config *config) {
    vector<float> result;
    for (int i = 1; i <=3; i++) {
        char key[40];
        sprintf(key, "recognition.mean%d", i);
        float value = config->GetConfig(key, -1);
        if (value < 0)
            break;
        result.push_back(value);
    }
    return result;
}*/

/*======================= CDNN recognition  ========================= */
CdnnRecog::CdnnRecog(string configPath, string modelDir, 
					bool multi_thread, bool is_encrypt): Recognition(is_encrypt) {
    bool bRet = _extractor.InitModels(configPath.c_str(), modelDir.c_str());
    if (!bRet)
    {
        cout << "Fail to load " << configPath << endl;
        exit(-1);
    }
    _multi_thread = multi_thread;

	_transformation = new CdnnTransformation();
}

CdnnRecog::CdnnRecog(const string& model_dir, 
					bool multi_thread, 
					bool is_encrypt) : Recognition(is_encrypt) {
	string config_file, real_model_dir;
	addNameToPath(model_dir, "/0.config", config_file);
	real_model_dir = (model_dir.back() == '$') ? model_dir.substr(0, model_dir.length()-1) : model_dir;

	// decrypt config file
	string cfg_content;
	int ret = getConfigContent(config_file, _is_encrypt, cfg_content);
	if(ret != 0) {
		cout << "fail to decrypt config file." << endl;
		exit(-1);
	}

	// initial model
    bool bRet = _extractor.InitModels(cfg_content, real_model_dir);
    if (!bRet)
    {
        cout << "Fail to load " << config_file << endl;
        exit(-1);
    }
    _multi_thread = multi_thread;

	_transformation = NULL;
	_transformation = new CdnnTransformation();
	if(_transformation == NULL) {
    	throw new runtime_error("can't initialize transformation in CdnnRecog");
	}
}


CdnnRecog::~CdnnRecog() {
	if(_transformation) {
		delete _transformation;
		_transformation = NULL;
	}
}

void CdnnRecog::recog_impl(const vector<cv::Mat>& faces,
                           const std::vector<AlignResult>& alignment,
                           std::vector<RecogResult>& results) {

    assert(faces.size() == alignment.size());
    results.clear();
    results.resize(faces.size());
    for (size_t idx = 0; idx < faces.size(); ++idx)
    {
        // prepare landmarks
		AlignResult transformed_alignment = {};
		Mat transformed_img;
		_transformation->transform(faces[idx], alignment[idx], transformed_img, transformed_alignment);

		vector<double> transformedLandmarks;
		cvtLandmarks(transformed_alignment.landmarks, transformedLandmarks);

        // prepare image
        IplImage ipl_image = transformed_img;

        //extract feature
        auto& feature = results[idx].face_feat;
        int bRet = _extractor.ExtractFeature(feature, &ipl_image, transformedLandmarks, _multi_thread);
        if (bRet != 0)
        {
            cerr << "Fail to ExtractFeature" << endl;
            feature.clear();
            exit(-1);
        }

    }

}


/////////////////////////////////////////////////////////////

void CdnnCaffeRecog::ParseConfigFile(const string& cfg_content,
									vector<string>& model_defs, 
									vector<string>& weight_files, 
									vector<string>& layer_names,
									vector<int>& patch_ids,
									vector<int>& patch_dims,
									bool* is_color) {
	stringstream ss_cfg(cfg_content);

    // multi_thread_ = false; // use the multi-thread version for feature extraction
    // multi_process_ = false; // use the multi-process version for feature extraction
    int num_patches;
    int patch_id;
    int color;
								
	int multi_thread_or_proc;
    ss_cfg >> color;
    *is_color = (color > 0);
    ss_cfg >> multi_thread_or_proc;
    ss_cfg >> num_patches;
    // if (multi_thread_or_proc == 1) {
    //     multi_thread_ = true;
    //     assert(num_patches > 1);
    // } else if (multi_thread_or_proc == 2) {
    //     multi_process_ = true;
    //     assert(num_patches > 1);
    // }
    // int num_models = (multi_thread_ || multi_process_ ) ? num_patches + 1 : 1;
    int num_models = 1;
    for (int i = 0; i < num_models; ++i) {
        string model_def, weight_file, layer_name;
        int patch_dim;
        ss_cfg >> model_def;
        ss_cfg >> weight_file;
        ss_cfg >> layer_name;
        // if (multi_process_)
        //     ss_cfg >> patch_dim;
        model_defs.push_back(model_def);
        weight_files.push_back(weight_file);
        layer_names.push_back(layer_name);
        patch_dims.push_back(patch_dim);
    }
    for (int i = 0; i < num_patches; i++) {
        ss_cfg >> patch_id;
        patch_ids.push_back(patch_id);
    }
}

CdnnCaffeRecog::CdnnCaffeRecog(const string& model_dir, 
								int gpu_id,
								bool is_encrypt, 
								int batch_size) : Recognition(is_encrypt) {
	string cfg_file;
	addNameToPath(model_dir, "/recog_cdnn_caffe.cfg", cfg_file);
	string cfg_content;
	int ret = getConfigContent(cfg_file, _is_encrypt, cfg_content);
	if(ret != 0) {
		cerr << "can't decrypt the recog_cdnn_caffe.cfg" << endl;
		return;
	}
	
    vector<string> model_defs, weight_files, layer_names;
    bool is_color;
    vector <int> patch_ids;
    vector <int> patch_dims;
	ParseConfigFile(cfg_content, model_defs, weight_files, layer_names, patch_ids, patch_dims, &is_color);
	assert(model_defs.size() == weight_files.size());
    assert(model_defs.size() == 1);
    assert(layer_names.size() == 1);
	for(size_t i = 0; i < model_defs.size(); ++i) {
		addNameToPath(model_dir, "/" + model_defs[i], model_defs[i]);
		addNameToPath(model_dir, "/" + weight_files[i], weight_files[i]);
	}
    
	_impl = new CaffeBatchWrapper(gpu_id, layer_names[0], batch_size,
        model_defs[0], weight_files[0], patch_ids, is_encrypt);

	_transformation = NULL;
	_transformation = new CdnnTransformation(); // use cdnn transformation temporarily
	if(_transformation == NULL) {
    	throw new runtime_error("can't initialize transformation in CdnnCaffeRecog");
	}
}


CdnnCaffeRecog::~CdnnCaffeRecog() {
    delete _impl;
	if(_transformation) {
		delete _transformation;
		_transformation = NULL;
	}
}

void CdnnCaffeRecog::recog_impl(const vector<cv::Mat>& faces,
                                const std::vector<AlignResult>& alignment,
                                std::vector<RecogResult>& results) {

    assert(faces.size() == alignment.size());
    results.clear();
    results.reserve(faces.size());
    vector<Mat> images;
    vector< vector<float> > landmarks;
    for (size_t idx = 0; idx < faces.size(); ++idx) {
        // prepare landmarks
        // prepare image
		AlignResult transformed_alignment = {};
		Mat transform_img;
		_transformation->transform(faces[idx], alignment[idx], transform_img, transformed_alignment);

		vector<double> transformedLandmark;
		cvtLandmarks(transformed_alignment.landmarks, transformedLandmark);
		vector<float> target_lmks(transformedLandmark.begin(), transformedLandmark.end());

        images.push_back(transform_img);
        landmarks.push_back(target_lmks);

        if (images.size() == _batch_size) {
            ProcessBatchAppend(images, landmarks, results);
            images.clear();
            landmarks.clear();
        }
        
    }
    if (images.size() > 0 && images.size() < _batch_size) {
        ProcessBatchAppend(images, landmarks, results);
    }
}

void CdnnCaffeRecog::ProcessBatchAppend(const vector<Mat> &images,
        vector< vector<float> > &landmarks,
        vector<RecogResult> &results) {
    vector< vector<float> > output;
    _impl->predict(images, landmarks, output);
    size_t base_idx = results.size();
    results.resize(base_idx + images.size());
    for (size_t i = 0; i < images.size(); i++) {
        auto& feature = results[base_idx + i].face_feat;
        feature.assign(output[i].begin(), output[i].end());
    }
}

/////////////////////////////////////////////////////////////////////
// ----------------------recogintion fusion-----------------------//
FuseRecog::FuseRecog(string model_dir, int gpu_id, bool multi_thread, 
					bool is_encrypt, int batch_size) : Recognition(is_encrypt) {
	string cdnn_model_dir, cdnn_caffe_model_dir;
	string cfg_file;
	addNameToPath(model_dir, "/recog_fuse.json", cfg_file);
	ParseConfigFile(cfg_file, cdnn_model_dir, cdnn_caffe_model_dir, fuse_weight_0, fuse_weight_1);

	string full_cdnn_model_dir, full_cdnn_caffe_model_dir;
	addNameToPath(model_dir, "/" + cdnn_model_dir, full_cdnn_model_dir);
	addNameToPath(model_dir, "/" + cdnn_caffe_model_dir, full_cdnn_caffe_model_dir);

	recog_0 = NULL;
	recog_0 = new CdnnRecog(full_cdnn_model_dir, multi_thread, is_encrypt);
	if(recog_0 == NULL) {
        cout << "cdnn in fuse recognition init failed!" << endl;
		return;
	}

  	recog_1 = NULL;
	recog_1 = new CdnnCaffeRecog(full_cdnn_caffe_model_dir, gpu_id, is_encrypt, batch_size);
    if(recog_1 == NULL) {
        cout << "cdnn_caffe in fuse recognition init failed!" << endl;
		return;
    }
}

void FuseRecog::ParseConfigFile(string cfg_file, string& cdnn_model_dir, string& cdnn_caffe_model_dir, float& cdnn_weight, float& cdnn_caffe_weight) {
	string cfg_content;
	int ret = getConfigContent(cfg_file, _is_encrypt, cfg_content);
	if(ret != 0 ) {
		cout << "fail to decrypt config file: " << cfg_file << endl;
		cdnn_model_dir.clear();
		cdnn_caffe_model_dir.clear();
		return;
	}

	Config fuse_cfg;
	if(!fuse_cfg.LoadString(cfg_content)) {
		cout << "fail to parse " << cfg_file << endl;
		cdnn_model_dir.clear();
		cdnn_caffe_model_dir.clear();
		return;
	}
	cdnn_model_dir = static_cast<string>(fuse_cfg.Value("cdnnFolder"));
	cdnn_weight = static_cast<float>(fuse_cfg.Value("cdnnWeight"));
	cdnn_caffe_model_dir = static_cast<string>(fuse_cfg.Value("cdnnCaffeFolder"));
	cdnn_caffe_weight = static_cast<float>(fuse_cfg.Value("cdnnCaffeWeight"));
}

FuseRecog::~FuseRecog() {
    delete []recog_0;
    recog_0 = NULL;
    
    delete []recog_1;
    recog_1 = NULL;
}


void FuseRecog::feature_combine(const RecogResult& result_0, const RecogResult& result_1, float weight_0, float weight_1, RecogResult& combined_result) {
    combined_result.face_feat.clear();
    if(result_0.face_feat.size() == 0 || result_1.face_feat.size() == 0) {
        cout << "one of feature is empty!" << endl;
        return;
    }

    for(size_t i = 0; i < result_0.face_feat.size(); ++i) {
        combined_result.face_feat.push_back(result_0.face_feat[i] * weight_0);
    }
    for(size_t i = 0; i < result_1.face_feat.size(); ++i) {
        combined_result.face_feat.push_back(result_1.face_feat[i] * weight_1);
    }
}

void FuseRecog::feature_combine(const vector<RecogResult>& results_0, const vector<RecogResult>& results_1, float weight_0, float weight_1, vector<RecogResult>& combined_results) {
    if(results_0.size() != results_1.size()) {
        cout << "results number not match!" << endl;
        combined_results.clear();
        return;
    }

    combined_results.resize(results_0.size());
    for(size_t i = 0; i < results_0.size(); ++i) {
        RecogResult cur_result = {};
        feature_combine(results_0[i], results_1[i], weight_0, weight_1, cur_result);
        combined_results[i] = cur_result;
    }
}

void FuseRecog::recog_impl(const vector<cv::Mat>& faces,
                                const std::vector<AlignResult>& alignment,
                                std::vector<RecogResult>& results) {
    vector<RecogResult> recog_0_result, recog_1_result;

    recog_0->recog(faces, alignment, recog_0_result,"NONE");
    recog_1->recog(faces, alignment, recog_1_result,"NONE");

    feature_combine(recog_0_result, recog_1_result, fuse_weight_0, fuse_weight_1, results);
}

/*====================== select recognizer ======================== */
/*------------
Recognition *create_recognition(const string & prefix) {
    Config *config = Config::instance();
    string type    = config->GetConfig<string>(prefix + "recognition", "cnn");
    if (type == "cnn") {
        vector<float> mean = config->GetConfigArr(prefix + "recognition.cnn.mean", vector<float> {128, 128, 128});
        string model_file   = config->GetConfig<string>(prefix + "recognition.cnn.model_file");
        string trained_file = config->GetConfig<string>(prefix + "recognition.cnn.trained_file");
        string layer_name   = config->GetConfig<string>(prefix + "recognition.cnn.layer_name");
        float pixel_scale   = config->GetConfig(prefix + "recognition.cnn.pixel_scale", 256.0f);
        bool use_GPU        = config->GetConfig(prefix + "recognition.cnn.use_GPU", true);
        int gpu_id          = config->GetConfig(prefix + "recognition.cnn.gpu_id", 0);
        return new CNNRecog(model_file, trained_file, layer_name, mean, pixel_scale, use_GPU, gpu_id);
    } else if (type == "lbp") {
        int radius    = config->GetConfig<int>(prefix + "recognition.lbp.radius", 1);
        int neighbors = config->GetConfig<int>(prefix + "recognition.lbp.neighbors", 8);
        int grid_x    = config->GetConfig<int>(prefix + "recognition.lbp.grid_x", 8);
        int grid_y    = config->GetConfig<int>(prefix + "recognition.lbp.grid_y", 8);
        return new LbpRecog(radius, neighbors, grid_x, grid_y);
    } else if (type == "cdnn") {
        string model_dir = config->GetConfig<string>(prefix + "recognition.cdnn.model_dir");
        bool multi_thread = config->GetConfig<bool>(prefix + "recognition.cdnn.multi_thread", true);
        return new CdnnRecog(model_dir, multi_thread);
    }
    else if (type == "cdnn_caffe") {
        string model_dir = config->GetConfig<string>(prefix + "recognition.cdnn_caffe.model_dir");
        int gpu_id         = config->GetConfig(prefix + "recognition.cdnn_caffe.gpu_id", 0);// -1 for CPU, 0~3 for GPU
        return new CdnnCaffeRecog(model_dir, gpu_id);
    } else if (type == "fusion") {
        string model_dir  = config->GetConfig<string>(prefix + "recognition.fusion.model_dir");
        int gpu_id        = config->GetConfig(prefix + "recognition.fusion.gpu_id", 0);// -1 for CPU, 0~3 for GPU
        bool multi_thread = config->GetConfig<bool>(prefix + "recognition.fusion.multi_thread", true);
        return new FuseRecog(model_dir, gpu_id, multi_thread);
    }
    throw new runtime_error("unknown recognition");
}
*/

Recognition *create_recognition_with_global_dir(const recog_method& method, 
											const string& global_dir,
											int gpu_id,
											bool multi_thread,
											bool is_encrypt,
											int batch_size) {
	string global_config_file;
	string tmp_model_dir = is_encrypt ? getEncryptModelDir() : getNonEncryptModelDir() ;	
	addNameToPath(global_dir, "/"+tmp_model_dir+"/"+getGlobalConfig(), global_config_file); 
	return create_recognition_with_config(method, global_config_file, gpu_id, multi_thread, is_encrypt, batch_size);
}
Recognition *create_recognition_with_config(const recog_method& method, 
										const string& config_file,
										int gpu_id,
										bool multi_thread,
										bool is_encrypt,
										int batch_size) {

const std::map<recog_method, std::string> recog_map {
	{recog_method::LBP, "LBP"},
	{recog_method::CNN, "CNN"},
	{recog_method::CDNN, "CDNN"},
	{recog_method::CDNN_CAFFE, "CDNN_CAFFE"},
	{recog_method::FUSION, "FUSION"}
};
	string recog_key = "FaceRecognition";
	string full_key = recog_key + "/" + recog_map.at(method);

	Config path_cfg;
	path_cfg.Load(config_file);
	string model_path = static_cast<string>(path_cfg.Value(full_key));
	if(model_path.empty()){
		throw new runtime_error(full_key + " not exist!");
	} else {
		return create_recognition(method, model_path, gpu_id, multi_thread, is_encrypt, batch_size);
	}
}
Recognition *create_recognition(const recog_method& method, 
								const string& model_dir,
								int gpu_id,
								bool multi_thread,
								bool is_encrypt,
								int batch_size) {
	switch(method) {
		case recog_method::FUSION: {
        	return new FuseRecog(model_dir, gpu_id, multi_thread, is_encrypt, batch_size);
			break;	
		}
		case recog_method::CDNN_CAFFE: {
        	return new CdnnCaffeRecog(model_dir, gpu_id, is_encrypt, batch_size);
			break;	
		}
		case recog_method::CDNN: {
        	return new CdnnRecog(model_dir, multi_thread, is_encrypt);
			break;
		}
		case recog_method::CNN: {
			throw new runtime_error("don't use cnn!");
			break;
		}
		case recog_method::LBP: {
			throw new runtime_error("don't use lbp!");
			break;
		}
		default:
    		throw new runtime_error("unknown recognition");
	}

	// if (method == "cdnn_caffe") {
    //     return new CdnnCaffeRecog(model_dir, gpu_id);
	// } else if (method == "cdnn") {
    //     return new CdnnRecog(model_dir, multi_thread);
	// } else if (method == "fusion") {
    //     return new FuseRecog(model_dir, gpu_id, multi_thread);
	// } else if (method == "lbp") {
	// 	throw new runtime_error("don't use lbp!");
	// } else if (method == "cnn") {
	// 	throw new runtime_error("don't use cnn!");
	// }
    // throw new runtime_error("unknown recognition");
}
}
