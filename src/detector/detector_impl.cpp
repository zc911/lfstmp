#include <detector/det_dlib.h>
#include <detector/det_rpn.h>
#include <detector/det_ssd.h>
#include <detector/det_fcn.h>
#include <stdexcept>
#include "dlib/image_processing.h"
#include "dlib/opencv.h"
#include "dlib/pixel.h"
#include "dlib_utils.h"
#include "caffe_interface.h"
#include "face_inf.h"
#include "dgface_utils.h"
#include "dgface_config.h"
#include <chrono>
using namespace cv;
using namespace std;
using namespace caffe;
namespace DGFace{

/*======================= Dlib detector ========================= */
DlibDetector::DlibDetector(int img_scale_max, int img_scale_min)
        : Detector(img_scale_max, img_scale_min, false),
        _detector(dlib::get_frontal_face_detector()) {
}

DlibDetector::~DlibDetector(void) {
}

void DlibDetector::detect_impl(const vector<Mat> &imgs, vector<DetectResult> &results) {
    for (size_t idx = 0; idx < imgs.size(); idx++) {
        Mat img = imgs[idx];
        if (img.channels() == 4)
            cvtColor(img, img, COLOR_BGRA2GRAY);
        else if (img.channels() == 3)
            cvtColor(img, img, COLOR_BGR2GRAY);

        assert(img.channels() == 1);
        dlib::array2d<unsigned char> dlib_img;
        dlib::assign_image(dlib_img, dlib::cv_image<unsigned char>(img));
        // dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
        vector< pair<double, dlib::rectangle> > dets;
        pyramid_up(dlib_img);
        _detector(dlib_img, dets);
        auto &boundingBox = results[idx].boundingBox;
        boundingBox.resize(dets.size());
        for (size_t i = 0; i < dets.size(); ++i) {
            auto &entry = boundingBox[i];
            auto &det   = dets[i];
            entry.first = det.first; //confidence 
			Rect m_box;
            dlib_rect2cv(det.second, m_box); //bbox
            m_box.x /= 2;
            m_box.y /= 2;
            m_box.width /= 2;
            m_box.height /= 2;

			Point2f box_center(m_box.x + m_box.width * 0.5, m_box.y + m_box.height * 0.5);
			Size2f box_size(m_box.width, m_box.height);
            entry.second = RotatedRect(box_center, box_size, 0);
        }
    }
}

/*======================= Rpn detector ========================== */
RpnDetector::RpnDetector(int img_scale_max,
        int   img_scale_min,
        const string& model_file,
        const string& trained_file,
        const string& layer_name_cls,
        const string& layer_name_reg,
        const vector <float> area,
        const vector <float> ratio,
        const vector <float> mean,
        const float  det_thresh,
        const size_t max_per_img,
        const size_t stride,
        const float  pixel_scale,
        const bool   use_GPU,
        const int    gpu_id)
         : Detector(img_scale_max, img_scale_min, false), _net(nullptr),
        _layer_name_cls(layer_name_cls), _layer_name_reg(layer_name_reg),
        _area(area), _ratio(ratio),
        _pixel_means(mean), _det_thresh(det_thresh), _max_per_img(max_per_img),
        _stride(stride), _pixel_scale(pixel_scale), _useGPU(use_GPU),_gpuid(gpu_id) {
    if (_useGPU)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    /* Load the network. */
    _net.reset(new Net<float>(model_file, TEST));
    _net->CopyTrainedLayersFrom(trained_file);

    Blob<float>* input_blob = _net->input_blobs()[0];
    _batch_size = input_blob->num();

    _num_channels  = input_blob->channels();
    CHECK(_num_channels == 3 || _num_channels == 1)
            << "Input layer should have 1 or 3 channels.";
}

RpnDetector::~RpnDetector(void) {
}

static bool mycmp(Bbox b1, Bbox b2)
{
    return b1.first > b2.first;
}

void RpnDetector::nms(vector<Bbox>& p, vector<bool>& deleted_mark, float threshold)
{
    sort(p.begin(), p.end(), mycmp);
    for(size_t i = 0; i < p.size(); i++)
    {
        if(deleted_mark[i]) continue;
        for(size_t j = i+1; j < p.size(); j++)
        {

            if(!deleted_mark[j])
            {
                cv::Rect intersect = p[i].second & p[j].second;
                float iou = intersect.area() * 1.0f / (p[i].second.area() + p[j].second.area() - intersect.area()); //p[j].second.area();
                if (iou > threshold)
                {
                    deleted_mark[j] = true;
                }
            }
        }
    }
}

void RpnDetector::detect_impl(const vector< cv::Mat > &imgs, vector<DetectResult> &results)
{
    if (!device_setted_) {
        Caffe::SetDevice(_gpuid);
        Caffe::set_mode(Caffe::GPU);
        device_setted_ = true;
    }
    results.resize(0);
    results.resize(imgs.size());
    int scale_num = _area.size() * _ratio.size();

    Blob<float>* input_blob = _net->input_blobs()[0];

    _image_size = Size(imgs[0].cols, imgs[0].rows);
    vector<int> shape = {static_cast<int>(imgs.size()), _num_channels, _image_size.height, _image_size.width};
    //vector<int> shape = {static_cast<int>(imgs.size()), 3, _image_size.height, _image_size.width};
    input_blob->Reshape(shape);
    _net->Reshape();
    float* input_data = input_blob->mutable_cpu_data();
    //cout<<_pixel_scale<<" "<<_pixel_means[0];
    for(size_t i = 0; i < imgs.size(); i++)
    {
        Mat sample;
        Mat img = imgs[i];
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
                        (float(sample.at<uchar>(row, col * sample.channels() + k)) - _pixel_means[k]) / _pixel_scale;
                }
            }
        }
    }
    _net->ForwardPrefilled();
    if(_useGPU)
    {
        cudaDeviceSynchronize();
    }

    auto cls = _net->blob_by_name(_layer_name_cls);
    auto reg = _net->blob_by_name(_layer_name_reg);
    assert(cls->channels() == scale_num * 2);
    assert(reg->channels() == scale_num * 4);
    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());
    const float* cls_cpu = cls->cpu_data();
    const float* reg_cpu = reg->cpu_data();

#ifdef SHOW_DEBUG
    cout << "[debug num, channels, height, width] " << cls->num() << " " << cls->channels() << " " << cls->height() << " " << cls->width() << endl;
    cout << "[scale_num]" << scale_num << endl;
    cout << "[debug stride, h,w] " << _stride << " " << cls->height()  << " " << cls->width()<< endl;
#endif
    vector<Bbox> vbbox;
    vector<bool> deleted_mark;
    vector<float> gt_ww, gt_hh;
    gt_ww.resize(scale_num);
    gt_hh.resize(scale_num);

    for(size_t i = 0; i < _area.size(); i++)
    {
        for(size_t j = 0; j < _ratio.size(); j++)
        {
            int index = i * _ratio.size() + j;
            gt_ww[index] = sqrt(_area[i] * _ratio[j]);
            gt_hh[index] = gt_ww[index] / _ratio[j];
        }
    }
    int cls_index = 0;
    int reg_index = 0;
    for (int img_idx = 0; img_idx < cls->num(); img_idx++)
    {
        vbbox.resize(0);
        for (int scale_idx = 0; scale_idx < scale_num; scale_idx++)
        {
            int skip = cls->height() * cls->width();
            for(int h = 0; h < cls->height(); h++)
            {
                for(int w = 0; w < cls->width(); w++)
                {
                    float confidence;
                    float rect[4] = {};
                    {
                        float x0 = cls_cpu[cls_index];
                        float x1 = cls_cpu[cls_index + skip];
                        float min_01 = min(x1, x0);
                        //if(h<5)
                        //    cout<<x0<<" "<<x1;
                        x0 -= min_01;
                        x1 -= min_01;
                        confidence = exp(x1) / (exp(x1) + exp(x0));
                    }
                    if( confidence > _det_thresh)
                    {
                        for(int j = 0; j < 4; j++)
                        {
                            rect[j] = reg_cpu[reg_index + j * skip];
                        }

                        float shift_x = w * _stride + _stride / 2.f - 1;
                        float shift_y = h * _stride + _stride / 2.f - 1;
                        rect[2] = exp(rect[2]) * gt_ww[scale_idx];
                        rect[3] = exp(rect[3]) * gt_hh[scale_idx];
                        rect[0] = rect[0] * gt_ww[scale_idx] - rect[2] / 2.f + shift_x;
                        rect[1] = rect[1] * gt_hh[scale_idx] - rect[3] / 2.f + shift_y;

                        Bbox bbox;
                        bbox.first   = confidence;
                        bbox.second  = Rect(rect[0], rect[1], rect[2], rect[3]);
                        bbox.second &= Rect(0, 0, _image_size.width, _image_size.height);
                        vbbox.push_back(bbox);
                        deleted_mark.push_back(false);
                    }

                    cls_index += 1;
                    reg_index += 1;
                }
            }
            cls_index += skip;
            reg_index += 3 * skip;
        }
        
        nms(vbbox, deleted_mark, 0.3);
        for(size_t i = 0; i < vbbox.size(); i++)
        {
            if(!deleted_mark[i])
            {
                auto &boundingBox = results[img_idx].boundingBox;

                ///////////////////////////////////////////////////
                // adjust bounding box

                const float h_rate = 0.30;
                const float w_rate = 0.15;
                
                Rect adjust_box = vbbox[i].second;
                adjust_box=adjust_box&Rect(1,1,imgs[img_idx].cols-1,imgs[img_idx].rows-1);
                float a_dist = vbbox[i].second.height * h_rate;

                adjust_box.y += a_dist;
                adjust_box.height -= a_dist;

                a_dist = vbbox[i].second.width * w_rate; 
                adjust_box.x += a_dist;
                adjust_box.width -= a_dist*2;

				RotatedBbox rot_bbox;
				Point2f box_center(adjust_box.x + adjust_box.width * 0.5, adjust_box.y + adjust_box.height * 0.5);
				Size2f box_size(adjust_box.width, adjust_box.height);
				rot_bbox.first = vbbox[i].first;
                rot_bbox.second = RotatedRect(box_center, box_size, 0);

                boundingBox.push_back(rot_bbox);
            }
        }
    }
    //cout<<endl;
}

/*======================= SSD detector ========================== */
SSDDetector::SSDDetector(int img_scale_max,
						int img_scale_min,
						const string& model_dir,
						int gpu_id,
						bool is_encrypt,
						int batch_size)
						:Detector(img_scale_max, img_scale_min, is_encrypt), 
						_net(nullptr), _gpuid(gpu_id) {
	device_setted_ = false;						
	if (_gpuid < 0) {
		_useGPU = false;
		Caffe::set_mode(Caffe::CPU);
	} else {
		_useGPU = true;
		Caffe::SetDevice(gpu_id);
		Caffe::set_mode(Caffe::GPU);
	}

	string cfg_file;
	addNameToPath(model_dir, "/det_ssd.json", cfg_file);	

	string deploy_file, model_file;
	// parse config file and read parameters(pixel_scale, det_thresh, mean)
	ParseConfigFile(cfg_file, deploy_file, model_file);
	addNameToPath(model_dir, "/" + deploy_file, deploy_file);
	addNameToPath(model_dir, "/" + model_file, model_file);

	string deploy_content, model_content;
	int ret = 0;
	ret = getFileContent(deploy_file, is_encrypt, deploy_content);
	if(ret != 0) {
		LOG(ERROR) << "failed decrypt " << deploy_file << endl;
		throw new runtime_error("decrypt failed!");
	}
	ret = getFileContent(model_file, is_encrypt, model_content);
	if(ret != 0) {
		LOG(ERROR) << "failed decrypt " << model_file << endl;
		throw new runtime_error("decrypt failed!");
	}

	/* Load the network. */
	_net.reset(new Net<float>(deploy_file, deploy_content, TEST));
	_net->CopyTrainedLayersFrom(model_file, model_content);

	//CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
	Blob<float>* input_blob = _net->input_blobs()[0];
	_batch_size = batch_size;
	if(input_blob->num() != batch_size) {
		vector<int> org_shape = input_blob->shape();
		org_shape[0] = batch_size;
		input_blob->Reshape(org_shape);
		_net->Reshape();
	} 

	_num_channels = input_blob->channels();
	CHECK(_num_channels == 3 || _num_channels == 1)
	   << "Input layer should have 1 or 3 channels.";
	if(_num_channels == 3) {
		if(_pixel_means.size() == 1) {
			_pixel_means.resize(_num_channels, _pixel_means[0]);
		}
	} else {
		CHECK(_num_channels == _pixel_means.size())
		   << "Mean vector size and input channels not match.";
	}
}

int SSDDetector::ParseConfigFile(const string& cfg_file, string& deploy_file, string& model_file) {
		
	Config ssd_cfg;
	if(!ssd_cfg.Load(cfg_file)) {
		cout << "fail to parse " << cfg_file << endl;
		deploy_file.clear();
		model_file.clear();
	}

	deploy_file = static_cast<string>(ssd_cfg.Value("deployfile"));
	model_file = static_cast<string>(ssd_cfg.Value("modelfile"));
	_pixel_scale = static_cast<float>(ssd_cfg.Value("pixel_scale"));
	_det_thresh = static_cast<float>(ssd_cfg.Value("det_thresh"));

	if(!static_cast<string>(ssd_cfg.Value("img_scale_max")).empty()) {
		_img_scale_max = static_cast<int>(ssd_cfg.Value("img_scale_max"));
	} else {
		_img_scale_max = 0;
	}
	if(!static_cast<string>(ssd_cfg.Value("img_scale_min")).empty()) {
		_img_scale_min = static_cast<int>(ssd_cfg.Value("img_scale_min"));
	} else {
		_img_scale_min = 0;
	}
	
	int mean_vec_size = static_cast<int>(ssd_cfg.Value("mean/Size"));
	CHECK(mean_vec_size == 3 || mean_vec_size == 1)
	   << "Mean vector should have 1 or 3 entries.";
	_pixel_means.resize(mean_vec_size);
	for(int i = 0; i < mean_vec_size; ++i) {
		_pixel_means[i] = static_cast<float>(ssd_cfg.Value("mean" + to_string(i)));
	}

	_bbox_shrink = static_cast<bool>(ssd_cfg.Value("bbox_shrink"));
	_use_deploy_input_size = static_cast<bool>(ssd_cfg.Value("use_deploy_input_size"));
}

/*
SSDDetector::SSDDetector(int   img_scale_max,
                        int   img_scale_min,
                        const string& model_file,
                        const string& trained_file,
                        const vector<float> mean,
                        const float det_thresh,
                        const float pixel_scale,
                        const bool  use_GPU,
                        const int   gpu_id)
                        :Detector(img_scale_max, img_scale_min, false), _net(nullptr), 
                        _pixel_means(mean), _det_thresh(det_thresh),
                        _pixel_scale(pixel_scale), _useGPU(use_GPU),_gpuid(gpu_id) {
   if (_useGPU) {
       Caffe::set_mode(Caffe::GPU);
       Caffe::SetDevice(gpu_id);
   }
   else {
       Caffe::set_mode(Caffe::CPU);
   }

   cout<<"loading "<<model_file<<endl;
   _net.reset(new Net<float>(model_file, TEST));
   cout<<"loading "<<trained_file<<endl;
   _net->CopyTrainedLayersFrom(trained_file);

   //CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
   Blob<float>* input_blob = _net->input_blobs()[0];
   _batch_size = input_blob->num();

   _num_channels = input_blob->channels();
   CHECK(_num_channels == 3 || _num_channels == 1)
       << "Input layer should have 1 or 3 channels.";
}
*/

SSDDetector::~SSDDetector(void){

}

void SSDDetector::detect_impl(const vector< cv::Mat > &imgs, vector<DetectResult> &results)
{
    if (!device_setted_) {
Caffe::SetDevice(_gpuid);
        Caffe::set_mode(Caffe::GPU);
        device_setted_ = true;
    }
    results.resize(0);
    results.resize(imgs.size());

    Blob<float>* input_blob = _net->input_blobs()[0];

	vector<Mat> resized_imgs(imgs.size());
	if(_use_deploy_input_size) {
		_image_size = Size(input_blob->width(), input_blob->height());
		for(size_t i = 0; i < imgs.size(); ++i) {
			cv::resize(imgs[i], resized_imgs[i], _image_size);
		}	
	} else {
    	_image_size = Size(imgs[0].cols, imgs[0].rows);
		for(size_t i = 0; i < imgs.size(); ++i) {
			resized_imgs[i] = imgs[i];
		}	
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
                        (float(sample.at<uchar>(row, col * sample.channels() + k)) - _pixel_means[k]) / _pixel_scale;
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
    //cout << "test place 5" << endl;

    const float* top_data = outputs[0]->mutable_cpu_data();
#ifdef SHOW_DEBUG
    cout << "[debug num, channels, height, width] " << outputs[0]->num() << " " << outputs[0]->channels() << " " << outputs[0]->height() << " " << outputs[0]->width() << endl;
#endif

    // vector<Bbox> &bboxes = results[i].boundingBox;
    // results[i].boundingBox.resize(outputs[0]->height);
	auto input_image_size = Size(imgs[0].cols, imgs[0].rows);
    for(int j = 0; j < outputs[0]->height(); j++) {
        int  img_id = top_data[j * 7];
        // int cls = top_data_win[j * 7 + 1];
        float score = top_data[j * 7 + 2];
        //float xmin = top_data[j * 7 + 3]*_image_size.width;
        //float ymin = top_data[j * 7 + 4]*_image_size.height; 
        //float xmax = top_data[j * 7 + 5]*_image_size.width; 
        //float ymax = top_data[j * 7 + 6]*_image_size.height;
        float xmin = top_data[j * 7 + 3]*input_image_size.width;
        float ymin = top_data[j * 7 + 4]*input_image_size.height; 
        float xmax = top_data[j * 7 + 5]*input_image_size.width; 
        float ymax = top_data[j * 7 + 6]*input_image_size.height;

        //cout << "img id: " << img_id << ", score:" << score << " " << xmin << " " << ymin << " " << xmax-xmin << " " << ymax-ymin << endl;
        if(score > _det_thresh){
            Bbox bbox;
            bbox.first = score;
            bbox.second.x = xmin;
            bbox.second.y = ymin;
            bbox.second.width = xmax-xmin;
            bbox.second.height = ymax-ymin;

            /////////////////////////////////////////
            // adjust the bounding box

            auto adjust_box = bbox.second;
			if(_bbox_shrink) {
				adjust_box=adjust_box&Rect(1,1,imgs[img_id].cols-1,imgs[img_id].rows-1);

				float a_dist = bbox.second.height * 0.40;

				adjust_box.y += a_dist;
				adjust_box.height -= a_dist;

				a_dist = bbox.second.width * 0.1; 
				adjust_box.x += a_dist;
				adjust_box.width -= a_dist*2;
			}

			RotatedBbox rot_bbox;
			Point2f box_center(adjust_box.x + adjust_box.width * 0.5, adjust_box.y + adjust_box.height * 0.5);
			Size2f box_size(adjust_box.width, adjust_box.height);

			rot_bbox.first = score;
            rot_bbox.second = RotatedRect(box_center, box_size, 0);

            //cout << "test place 6" << endl;
            results[img_id].boundingBox.push_back(rot_bbox);
        }
    }

    //cout << "test place 7" << endl;
    memset(_net->output_blobs()[0]->mutable_cpu_data(), 0, sizeof(*_net->output_blobs()[0]->mutable_cpu_data()) * _net->output_blobs()[0]->count());
}

/*====================== fcn detector ======================== */
FcnDetector::FcnDetector(int img_scale_max, 
			int img_scale_min, 
			const std::string& model_dir, 
			int gpu_id, 
			bool is_encrypt, 
			int batch_size)
                	: Detector(img_scale_max, img_scale_min, is_encrypt),
			_min_det_face_size(24), _max_det_face_size(-1), 
			_min_scale_face_to_img(0.1), _gpuid(gpu_id),
                        _batch_size(batch_size) {

    _device_setted_ = false;						
    if (_gpuid < 0) {
	_useGPU = false;
	Caffe::set_mode(Caffe::CPU);
    } else {
	_useGPU = true;
	Caffe::SetDevice(gpu_id);
	Caffe::set_mode(Caffe::GPU);
    }

    string cfg_file;
    string deploy_file, model_file;
    addNameToPath(model_dir, "/det_fcn.json", cfg_file);	
    
    ParseConfigFile(cfg_file, deploy_file, model_file);
    
    string full_deploy_file, full_model_file;
    addNameToPath(model_dir, "/" + deploy_file, full_deploy_file);
    addNameToPath(model_dir, "/" + model_file, full_model_file);
    
    string full_deploy_content, full_model_content;
    int ret = 0;
    ret = getFileContent(full_deploy_file, is_encrypt, full_deploy_content);
    if(ret != 0) {
    	cout << "failed decrypt " << full_deploy_file << endl;
    	throw new runtime_error("decrypt failed!");
    }
    ret = getFileContent(full_model_file, is_encrypt, full_model_content);
    if(ret != 0) {
    	cout << "failed decrypt " << full_model_file << endl;
    	throw new runtime_error("decrypt failed!");
    }

    /* Load the network. */
    _net.reset(new Net<float>(full_deploy_file, full_deploy_content, TEST));
    LOG(INFO) << "Load file " << full_deploy_file;

    _net->CopyTrainedLayersFrom(full_model_file, full_model_content);
    LOG(INFO) << "Load file " << full_model_file;

    _pryd_db = new db::PyramidDenseBox(_min_det_face_size, _max_det_face_size, _min_scale_face_to_img);
    assert(_pryd_db != nullptr);

    LOG(INFO) << "FCN Initialized";
}

void FcnDetector::ParseConfigFile(string cfg_file, string& deploy_file, string& model_file) {

    string cfg_content;
    int ret = getConfigContent(cfg_file, false, cfg_content);
    if(ret != 0) {
	LOG(ERROR) << "fail to decrypt " << cfg_file;
    	deploy_file.clear();
    	model_file.clear();
    	return;
    }
    
    Config fcn_cfg;
    if(!fcn_cfg.LoadString(cfg_content)) {
    	LOG(ERROR) << "fail to parse " << cfg_file;
    	deploy_file.clear();
    	model_file.clear();
    	return;
    }
    
    deploy_file = static_cast<string>(fcn_cfg.Value("deployfile"));
    model_file = static_cast<string>(fcn_cfg.Value("modelfile"));
    
    if(!static_cast<string>(fcn_cfg.Value("img_scale_max")).empty()) {
    	_img_scale_max = static_cast<int>(fcn_cfg.Value("img_scale_max"));
    } else {
    	_img_scale_max = 0;
    }
    if(!static_cast<string>(fcn_cfg.Value("img_scale_min")).empty()) {
    	_img_scale_min = static_cast<int>(fcn_cfg.Value("img_scale_min"));
    } else {
    	_img_scale_min = 0;
    }
    
    if(!static_cast<string>(fcn_cfg.Value("batch_img_height")).empty()) {
        _batch_img_height = static_cast<int>(fcn_cfg.Value("batch_img_height"));
    }
    if(!static_cast<string>(fcn_cfg.Value("batch_img_width")).empty()) {
        _batch_img_width = static_cast<int>(fcn_cfg.Value("batch_img_width"));
    }
    
    _min_det_face_size = static_cast<int>(fcn_cfg.Value("min_det_face_size"));
    _max_det_face_size = static_cast<int>(fcn_cfg.Value("max_det_face_size"));
    _min_scale_face_to_img = static_cast<float>(fcn_cfg.Value("min_scale_face_to_img"));
}

FcnDetector::~FcnDetector() {
    if(_pryd_db) {
    	delete _pryd_db;
    }
}

RotatedRect cvtDetectInfoToRotatedRect(const DetectedFaceInfo& det_info) {

    vector<float> rbox(5,0);
    float degree_rad = (float)det_info.degree*CV_PI/180;
    float cos_degree = cos(degree_rad),
          sin_degree = sin(degree_rad);

    // cout << "cvtDetectInfoToRotatedRect before: x = " << det_info.left << ", y = " << det_info.top << ", width = " << det_info.width << ", height = " << det_info.height << endl;

    assert( rbox.size() == 5);
    rbox[0] = det_info.left + cos_degree*det_info.width/2 - sin_degree*det_info.height/2;
    rbox[1] = det_info.top  + sin_degree*det_info.width/2 + cos_degree*det_info.height/2;
    rbox[2] = det_info.width;
    rbox[3] = det_info.height;
    rbox[4] = det_info.degree;

    // cout << "cvtDetectInfoToRotatedRect: x = " << rbox[0] << ", y = " << rbox[1] << ", width = " << rbox[2] << ", height = " << rbox[3] << "degree: " << rbox[4] <<  endl;

    return RotatedRect(Point2f(rbox[0], rbox[1]), Size2f(rbox[2], rbox[3]), rbox[4]);
}

RotatedBbox cvtPyrdBoxToRotatedBbox(const db::RotateBBox<float>& rBBox) {
    RotatedBbox output_bbox;
    output_bbox.first = rBBox.score;
    float bbox_width = std::sqrt( (rBBox.rt_x - rBBox.lt_x)*(rBBox.rt_x - rBBox.lt_x)
	         		+ (rBBox.rt_y - rBBox.lt_y)*(rBBox.rt_y - rBBox.lt_y) );
    output_bbox.second.size.width = bbox_width;
    output_bbox.second.size.height = bbox_width;

    float degree_rad  = std::atan2( rBBox.rt_y-rBBox.lt_y, rBBox.rt_x-rBBox.lt_x );
    float cos_degree = std::cos(degree_rad),
          sin_degree = std::sin(degree_rad);

    float center_x = rBBox.lt_x + (cos_degree - sin_degree) * bbox_width /2;
    float center_y = rBBox.lt_y + (sin_degree + cos_degree) * bbox_width /2;
    output_bbox.second.center.x = center_x;
    output_bbox.second.center.y = center_y;

    float degree = degree_rad * 180 / CV_PI;
    output_bbox.second.angle = degree>0 ? (degree+0.5) : (degree-0.5);
    
    return output_bbox;
}

void FcnDetector::detect_impl(const vector< cv::Mat > &imgs, vector<DetectResult> &results) {
    results.clear();
    vector<Mat> batch_imgs;
    for(size_t i = 0; i < imgs.size(); ++i) {
        batch_imgs.push_back(imgs[i]);

	if(batch_imgs.size() == _batch_size) {
            vector<DetectResult> batch_results;
	    detect_batch(batch_imgs, batch_results);

	    results.insert(results.end(), batch_results.begin(), batch_results.end());
	    batch_imgs.clear();
	}
    }
    if(batch_imgs.size() > 0) {
        CHECK(batch_imgs.size() <= _batch_size) << "actual batch size too large";

        vector<DetectResult> batch_results;
        detect_batch(batch_imgs, batch_results);

        results.insert(results.end(), batch_results.begin(), batch_results.end());
	batch_imgs.clear();
    }
}

void FcnDetector::detect_batch(const vector< cv::Mat > &imgs, vector<DetectResult> &results) {
    results.resize(imgs.size());
    
    vector<vector< db::RotateBBox<float> > > rotateFaces;
    _pryd_db->predictPyramidDenseBox(_net, imgs, rotateFaces);

    for(size_t i = 0; i < results.size(); ++i) {
	const auto& one_img_rotFaces = rotateFaces[i];
    	for(size_t j = 0; j < one_img_rotFaces.size(); ++j) {
	    results[i].boundingBox.push_back(cvtPyrdBoxToRotatedBbox(one_img_rotFaces[j]));
	}
    }
}
// void FcnDetector::detect_impl(const vector< cv::Mat > &imgs, vector<DetectResult> &results) {
//     results.resize(imgs.size());
//     
//     vector<vector< db::RotateBBox<float> > > rotateFaces;
//     _pryd_db->predictPyramidDenseBox(_net, imgs, rotateFaces);
// 
//     for(size_t i = 0; i < results.size(); ++i) {
// 	const auto& one_img_rotFaces = rotateFaces[i];
//     	for(size_t j = 0; j < one_img_rotFaces.size(); ++j) {
// 	    results[i].boundingBox.push_back(cvtPyrdBoxToRotatedBbox(one_img_rotFaces[j]));
// 	}
//     }
// }

// #define FCN_BATCH
// #ifndef FCN_BATCH
// void FcnDetector::detect_impl(const vector< cv::Mat > &imgs, vector<DetectResult> &results)
// {
//     results.resize(imgs.size());
//     for (size_t i = 0; i < imgs.size(); ++i) {
//         // prepare image
//         Mat img_copy = imgs[i].clone();
//         IplImage ipl_img = img_copy;
//         // prepare detection results vector
//         vector<DetectedFaceInfo> detectInfos;
// 
//         // detect
//         // _fcn_detecror->Detect(&ipl_img, _param, detectInfos);
//         _fcn_detecror->Detect(&ipl_img, _param, detectInfos, 
// 						_min_det_face_size, _max_det_face_size, _min_scale_face_to_img);
// 
//         // convert results
//         for (size_t result_idx = 0; result_idx < detectInfos.size(); ++result_idx) {
//             RotatedRect rot_rect = cvtDetectInfoToRotatedRect(detectInfos[result_idx]);
//             auto confidence = detectInfos[result_idx].conf;
//             results[i].boundingBox.push_back(make_pair(static_cast<float>(confidence),rot_rect));
//         }
// 
//     }
// 
// }
// #else
// void FcnDetector::detect_impl(const vector< cv::Mat > &imgs, vector<DetectResult> &results)
// {
//     results.resize(imgs.size());
// 
// 	// prepare image
// 	vector<IplImage*> ipl_imgs;
//     for (size_t i = 0; i < imgs.size(); ++i) {
// 		IplImage ipl_tmp = IplImage(imgs[i]);
// 		IplImage* ipl_store = cvCreateImage(cvSize(imgs[i].cols, imgs[i].rows),8,imgs[i].channels());
// 		cvCopy(&ipl_tmp, ipl_store);
// 		ipl_imgs.push_back(ipl_store);
// 	}
// 	
// 	vector<vector<DetectedFaceInfo> > detectInfos;
//     // detect
//     _fcn_detecror->Detect(ipl_imgs, _param, detectInfos, 
// 				     _min_det_face_size, _max_det_face_size, _min_scale_face_to_img);
// 
//     // convert results
//     for (size_t i = 0; i < imgs.size(); ++i) {
// 		const auto& one_img_detectInfo = detectInfos[i];
//         for (size_t result_idx = 0; result_idx < one_img_detectInfo.size(); ++result_idx) {
//             RotatedRect rot_rect = cvtDetectInfoToRotatedRect(one_img_detectInfo[result_idx]);
//             auto confidence = one_img_detectInfo[result_idx].conf;
//             results[i].boundingBox.push_back(make_pair(static_cast<float>(confidence),rot_rect));
//         }
// 	}
// 	
//     for (size_t i = 0; i < imgs.size(); ++i) {
// 			cvReleaseImage(&(ipl_imgs[i]));
// 	}
// }
// #endif

/*====================== select detector ======================== */

Detector *create_detector_with_global_dir(const det_method& method, 
					const string& global_dir,
					int gpu_id, 
					bool is_encrypt, 
					int batch_size) {

const std::map<det_method, std::string> det_map {
	{det_method::DLIB, "DLIB"},
	{det_method::SSD, "SSD"},
	{det_method::RPN, "RPN"},
	{det_method::FCN, "FCN"}
};
	string det_key = "FaceDetector";
	string full_key = det_key + "/" + det_map.at(method);

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

		return create_detector(method, model_path, gpu_id, is_encrypt, batch_size);
	}
}

Detector *create_detector_with_config(const det_method& method, 
									const string& config_file,
									int gpu_id, 
									bool is_encrypt, 
									int batch_size) {
const std::map<det_method, std::string> det_map {
	{det_method::DLIB, "DLIB"},
	{det_method::SSD, "SSD"},
	{det_method::RPN, "RPN"},
	{det_method::FCN, "FCN"}
};
	string det_key = "FaceDetector";
	string full_key = det_key + "/" + det_map.at(method);

	Config path_cfg;
	path_cfg.Load(config_file);
	string model_path = static_cast<string>(path_cfg.Value(full_key));
	if(model_path.empty()){
		throw new runtime_error(full_key + " not exist!");
	} else {
		return create_detector(method, model_path, gpu_id, is_encrypt, batch_size);
	}

}
Detector *create_detector(const det_method& method, 
							const string& model_dir, 
							int gpu_id, 
							bool is_encrypt, 
							int batch_size) {
	int img_scale_max = 720;
	int img_scale_min = 240;
	switch(method) {
		case det_method::FCN: {
			return new FcnDetector(img_scale_max, img_scale_min, model_dir, gpu_id, is_encrypt, batch_size);
			break;
		}
		case det_method::DLIB: {
			throw new runtime_error("don't use dlib!");
			break;
		}
		case det_method::RPN: {
			throw new runtime_error("don't use rpn!");
			break;
		}
		case det_method::SSD: {
			return new SSDDetector(img_scale_max, img_scale_min, model_dir, gpu_id, is_encrypt, batch_size);
			break;
		}
		default:
			throw new runtime_error("unknown detector");
	}
	// if(method == "fcn") {
    //     return new FcnDetector(img_scale_max, img_scale_min, model_dir, gpu_id);
	// } else if(method == "ssd") {
	// 	throw new runtime_error("don't use ssd!");
	// } else if(method == "rpn") {
	// 	throw new runtime_error("don't use rpn!");
	// } else if (method == "dlib") {
	// 	throw new runtime_error("don't use dlib!");
    //     return new DlibDetector(img_scale_max, img_scale_min);
	// }
    // throw new runtime_error("unknown detector");

}

}
