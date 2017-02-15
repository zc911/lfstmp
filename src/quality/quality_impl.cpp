#include <quality/qual_blurm.h>
#include <quality/qual_lenet_blur.h>
#include <quality/qual_frontalm.h>
#include <quality/qual_posem.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <stdint.h>
#include "caffe_interface.h"
#include "dgface_utils.h"
#include "dgface_config.h"

using namespace cv;
using namespace std;
using namespace caffe;

namespace DGFace{

/*======================= blur_metric quality ========================= */
BlurMQuality::BlurMQuality(void) {
}

BlurMQuality::~BlurMQuality(void) {
}

float BlurMQuality::blur_metric(const Mat &image, short *sobelTable)
{
    int i, j, mul;
    int width  = image.cols;
    int height = image.rows;
    const uint8_t *data = image.ptr();

    for(i = 1, mul = i * width; i < height - 1; i++, mul += width)
          for(j = 1; j < width - 1; j++)
            sobelTable[mul+j] = abs(data[mul+j-width-1] + 2*data[mul+j-1] + data[mul+j-1+width] -
                        data[mul+j+1-width] - 2*data[mul+j+1] - data[mul+j+width+1]);

    for(i = 1, mul = i*width; i < height - 1; i++, mul += width)
          for(j = 1; j < width - 1; j++)
            if(sobelTable[mul+j] < 50/* || sobelTable[mul+j] <= sobelTable[mul+j-1] ||\
                        sobelTable[mul+j] <= sobelTable[mul+j+1]*/) sobelTable[mul+j] = 0;
    int totLen = 0;
    int totCount = 1;

    unsigned char suddenThre = 50;
    unsigned char sameThre = 3;

    for(i = 1, mul = i*width; i < height - 1; i++, mul += width)
    {
        for(j = 1; j < width - 1; j++)
        {
            if(sobelTable[mul+j])
            {
                int   count = 0;
                int      t;
                unsigned char tmpThre = 5;
                unsigned char max = data[mul+j] > data[mul+j-1] ? 0 : 1;

                for(t = j; t > 0; t--)
                {
                    count++;
                    if(abs(data[mul+t] - data[mul+t-1]) > suddenThre)
                        break;

                    if(max && data[mul+t] > data[mul+t-1])
                        break;
                   

                    if(!max && data[mul+t] < data[mul+t-1])
                        break;

                    int tmp = 0;
                    for(int s = t; s > 0; s--)
                    {
                        if(abs(data[mul+t] - data[mul+s]) < sameThre)
                        {
                            tmp++;
                            if (tmp <= tmpThre)
                                continue;
                        }
                        break;
                    }

                    if(tmp > tmpThre) break;
                }

                max = data[mul+j] > data[mul+j+1] ? 0 : 1;

                for(t = j; t < width; t++)
                {
                    count++;
                    if(abs(data[mul+t] - data[mul+t+1]) > suddenThre)
                        break;

                    if((max && data[mul+t] > data[mul+t+1]) || (!max && data[mul+t] < data[mul+t+1]))
                        break;

                    int tmp = 0;
                    for(int s = t; s < width; s++)
                    {
                        if(abs(data[mul+t] - data[mul+s]) < sameThre)
                        {
                            tmp++;
                            if (tmp <= tmpThre)
                                continue;
                        }
                        break;
                    }
                    if(tmp > tmpThre) break;
                }
                count--;
                totCount++;
                totLen += count;
            }
        }
    }
    float result = static_cast<float>(totLen)/totCount;
    return result;
}

float BlurMQuality::quality(const Mat &image) {
    Mat sample;
    if (image.channels() == 3)
        cvtColor(image, sample, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cvtColor(image, sample, CV_BGRA2GRAY);
    unique_ptr<short> sobelTable(new short[sample.cols * sample.rows]);
    return blur_metric(sample, sobelTable.get());
}


/*======================= LenetBlur quality ============================*/

const int LenetBlurQuality::kBatchSize = 1;
int LenetBlurQuality::ParseConfigFile(const std::string& cfg_file, std::string& deploy_file,
			std::string& model_file) {
	Config ssd_cfg;
	if (!ssd_cfg.Load(cfg_file)) {
		cout << "fail to parse " << cfg_file << endl;
		deploy_file.clear();
		model_file.clear();
	}

	deploy_file = static_cast<string>(ssd_cfg.Value("deployfile"));
	model_file = static_cast<string>(ssd_cfg.Value("modelfile"));
	return 0;
}

LenetBlurQuality::LenetBlurQuality(const std::string& model_dir, int gpu_id, bool is_encrypt, int batch_size):_gpuid(gpu_id), _net(nullptr) {
	if (_gpuid < 0) {
		_useGPU = false;
		Caffe::set_mode(Caffe::CPU);
	} else {
		_useGPU = true;
		Caffe::SetDevice(gpu_id);
		Caffe::set_mode(Caffe::GPU);
	}
	string cfg_file;
	addNameToPath(model_dir, "/blur_lenet.json", cfg_file);
	string deploy_file, model_file;
	// parse config file and read parameters(pixel_scale, det_thresh, mean)
	ParseConfigFile(cfg_file, deploy_file, model_file);
	addNameToPath(model_dir, "/" + deploy_file, deploy_file);
	addNameToPath(model_dir, "/" + model_file, model_file);
	string deploy_content, model_content;
	int ret = 0;
	ret = getFileContent(deploy_file, is_encrypt, deploy_content);
	if (ret != 0) {
		LOG(ERROR) << "failed decrypt " << deploy_file << endl;
		throw new runtime_error("decrypt failed!");
	}
	ret = getFileContent(model_file, is_encrypt, model_content);
	if (ret != 0) {
		LOG(ERROR) << "failed decrypt " << model_file << endl;
		throw new runtime_error("decrypt failed!");
	}
	/* Load the network. */
	_net.reset(new Net<float>(deploy_file, deploy_content, TEST));
	_net->CopyTrainedLayersFrom(model_file, model_content);
	//CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
	Blob<float>* input_blob = _net->input_blobs()[0];
	_batch_size = batch_size;
	if (input_blob->num() != _batch_size) {
		vector<int> org_shape = input_blob->shape();
		org_shape[0] = _batch_size;
		input_blob->Reshape(org_shape);
		_net->Reshape();
	}
	_num_channels = input_blob->channels();
	CHECK(_num_channels == 3 || _num_channels == 1)
			<< "Input layer should have 1 or 3 channels.";
//	_pixel_means.push_back(128);
	for (int i=0; i<3; i++ ) {
		_pixel_means.push_back(128);
		printf("_pixel_means[%d] = %f\n", i, _pixel_means[i]);
	}
}

LenetBlurQuality::~LenetBlurQuality() {

}

float LenetBlurQuality::quality(const Mat &image) {
	Mat resized_img;
	Blob<float>* input_blob = _net->input_blobs()[0];
	_image_size = Size(input_blob->width(), input_blob->height());
	resize(image, resized_img, _image_size);
	vector<int> shape = { 1, _num_channels,
			_image_size.height, _image_size.width };
	input_blob->Reshape(shape);
	_net->Reshape();
	float* input_data = input_blob->mutable_cpu_data();
	Mat sample;
	Mat img = resized_img;
	// images from the same batch should have the same size
	assert(img.rows == _image_size.height && img.cols == _image_size.width);
	if (img.channels() == 3 && _num_channels == 1)
		cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && _num_channels == 1)
		cvtColor(img, sample, CV_BGRA2GRAY);
	else if (img.channels() == 4 && _num_channels == 3)
		cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && _num_channels == 3)
		cvtColor(img, sample, CV_GRAY2BGR);
	else
		sample = img;
	for (int k = 0; k < sample.channels(); k++) {
		size_t channel_off = k * sample.rows * sample.cols;
		for (int row = 0; row < sample.rows; row++) {
			size_t row_off = row * sample.cols;
			for (int col = 0; col < sample.cols; col++) {
				input_data[channel_off + row_off + col] = (float(
						sample.at<uchar>(row, col * sample.channels() + k))
						- _pixel_means[k]) / 255.0;
			}
		}
	}
	_net->ForwardPrefilled();
	if (_useGPU) {
		cudaDeviceSynchronize();
	}
	vector<Blob<float>*> outputs;
	outputs.clear();
	for (int i = 0; i < _net->num_outputs(); i++) {
		Blob<float>* output_layer = _net->output_blobs()[i];
		outputs.push_back(output_layer);
	}
	const float* top_data = outputs[0]->mutable_cpu_data();
	return top_data[0];
}

void LenetBlurQuality::quality(const std::vector<cv::Mat> &imgs, std::vector<float> &results) {
	results.resize(0);
	results.resize(imgs.size());

	Blob<float>* input_blob = _net->input_blobs()[0];

	vector<Mat> resized_imgs(imgs.size());
	_image_size = Size(input_blob->width(), input_blob->height());
	for (size_t i = 0; i < imgs.size(); ++i) {
		cv::resize(imgs[i], resized_imgs[i], _image_size);
	}

	vector<int> shape = {static_cast<int>(imgs.size()), _num_channels, _image_size.height, _image_size.width};
	input_blob->Reshape(shape);
	_net->Reshape();
	float* input_data = input_blob->mutable_cpu_data();
	//cout<<_pixel_scale<<" "<<_pixel_means[0];
	for(size_t i = 0; i < resized_imgs.size(); i++)
	{
	    Mat sample;
	    Mat img = resized_imgs[i];
	    // images from the same batch should have the same size
	    assert(img.rows == _image_size.height && img.cols == _image_size.width);
	    if (img.channels() == 3 && _num_channels == 1)
	        cvtColor(img, sample, CV_BGR2GRAY);
	    else if (img.channels() == 4 && _num_channels == 1)
	        cvtColor(img, sample, CV_BGRA2GRAY);
	    else if (img.channels() == 4 && _num_channels == 3)
	        cvtColor(img, sample, CV_BGRA2BGR);
	    else if (img.channels() == 1 && _num_channels == 3)
	        cvtColor(img, sample, CV_GRAY2BGR);
	    else
	        sample = img;

	    size_t image_off = i * sample.channels() * sample.rows * sample.cols;
	    for(int k = 0; k < sample.channels(); k++)
	    {
	        size_t channel_off = k * sample.rows * sample.cols;
	        for(int row = 0; row < sample.rows; row++)
	        {
	            size_t row_off = row * sample.cols;
	            for(int col = 0; col < sample.cols; col++)
	            {
	                input_data[image_off + channel_off + row_off + col] =
	                    (float(sample.at<uchar>(row, col * sample.channels() + k)) - _pixel_means[k]) / 255.0;
	            }
	        }
	    }
	}
	_net->ForwardPrefilled();
	if(_useGPU) {
	    cudaDeviceSynchronize();
	}
	/* Copy the output layer to a std::vector */
	vector<Blob<float>* > outputs;
	outputs.clear();
	for(int i = 0; i < _net->num_outputs(); i++) {
	    Blob<float>* output_layer = _net->output_blobs()[i];
	    outputs.push_back(output_layer);
	}
	const float* top_data = outputs[0]->mutable_cpu_data();
	printf("batch_size = %d\n", _batch_size);
	for ( int i=0; i<_batch_size; i++ ) {
		results[i] = top_data[i];
	}
	memset(_net->output_blobs()[0]->mutable_cpu_data(), 0, sizeof(*_net->output_blobs()[0]->mutable_cpu_data()) * _net->output_blobs()[0]->count());
}

/*======================= Frontal face quality ========================= */
FrontalMQuality::FrontalMQuality(void) : _detector(new DlibDetector(60, 45)) {
}

FrontalMQuality::~FrontalMQuality(void) {
    delete _detector;
}

float FrontalMQuality::quality(const Mat &image) {
    float frontal_score = 0.f;
    vector<DetectResult> fine_det_results;
    vector<Mat> images;
    images.push_back(image);
    
    _detector->detect(images, fine_det_results);
    if (fine_det_results[0].boundingBox.size()) {
        frontal_score = fine_det_results[0].boundingBox[0].first; // dlib det score
    }
    return frontal_score;
}

/*====================== pose quality ======================== */
PoseQuality::PoseQuality(void) {}

PoseQuality::~PoseQuality(void) {}

vector<float> PoseQuality::quality(const AlignResult &align_result) {
		//calculating the head pose
		Mat_<float> s(align_result.landmarks.size()*2,1);

		for(size_t i=0; i<align_result.landmarks.size(); ++i)
		{       
			s(i,0) = align_result.landmarks[i].x;
			s(i+align_result.landmarks.size(),0) = align_result.landmarks[i].y;
		}       
		HeadPose pose;
		EstimateHeadPose(s,pose);   //head pose estimation 

		vector<float> pose_angles;
		pose_angles.resize(sizeof(pose.angles)/sizeof(pose.angles[0]));
        pose_angles[0] = pose.angles[0];//pitch
        pose_angles[1] = pose.angles[1];//yaw
        pose_angles[2] = pose.angles[2];//roll
		return pose_angles;
}

/*====================== select detector ======================== */
/*
Quality *create_quality(const string &prefix) {
    Config *config    = Config::instance();
    string type       = config->GetConfig<string>(prefix + "quality", "blurm");

    if (type == "blurm")
        return new BlurMQuality();
    else if (type == "frontalm")
        // create dlib frontal face detector
        return new FrontalMQuality();
	else if (type == "posem")
	    // create pose estimation
		return new PoseQuality();
    throw new runtime_error("unknown quality measure");
}
*/
Quality *create_quality(const quality_method& method, const std::string& model_dir, int gpu_id, bool is_encrypt, int batch_size) {
	switch(method) {
		case BLURM: {
        	return new BlurMQuality();
			break;
		}
		case LENET_BLUR:
			return new LenetBlurQuality(model_dir, gpu_id, is_encrypt, batch_size);
			break;
		case FRONT: {
        	return new FrontalMQuality();
			break;
		}
		case POSE: {
			return new PoseQuality();
			break;
		}
		default:
			throw new runtime_error("unknown quality measure");
	}
}

Quality *create_quality_with_global_dir(const quality_method& method, const std::string& global_dir,
						int gpu_id,	bool is_encrypt, int batch_size) {
	const std::map<quality_method, std::string> quality_map {
		{quality_method::BLURM, "BLURM"},
		{quality_method::LENET_BLUR, "LENET_BLUR"},
		{quality_method::FRONT, "FRONT"},
		{quality_method::POSE, "POSE"}
	};
	string quality_key = "FaceQuality";
	string full_key = quality_key + "/" + quality_map.at(method);

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

		return create_quality(method, model_path, gpu_id, is_encrypt, batch_size);
	}
}
}
