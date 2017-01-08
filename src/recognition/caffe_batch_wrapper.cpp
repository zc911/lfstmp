#include <cassert>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <recognition/caffe_batch_wrapper.h>
#include "dgface_utils.h"
#define SAY(format, ...)    \
    fprintf(stderr, "%s(%s:%i): \t" format "\n",    \
         __func__, __FILE__, __LINE__, __VA_ARGS__)
using namespace caffe;
using namespace cv;
using namespace std;
using namespace DGFace;
enum face_patch {main_face = 0, left_eye, right_eye, nose, mouth, middle_eye, left_mouth, right_mouth, mini_face, left_brow, right_brow, middle_brow};

CaffeBatchWrapper::CaffeBatchWrapper(int deviceId, const string &layer_name, int batch_size,
        const string &model_def, const string &weights, const vector<int> &patch_ids, bool is_encrypt)
        : _batch_size(batch_size), _input_blob(NULL), _output_blob(NULL),
          _patch_ids(patch_ids) {
	Caffe::SetDevice(deviceId);
	Caffe::set_mode(Caffe::GPU);

	if(is_encrypt) {
		string deploy_content, weight_content;
		int ret = 0;
		ret = getFileContent(model_def, is_encrypt, deploy_content);
		if(ret != 0) {
			cout << "failed decrypt " << model_def << endl;
			throw new runtime_error("decrypt failed!");
		}
		ret = getFileContent(weights, is_encrypt, weight_content);
		if(ret != 0) {
			cout << "failed decrypt " << weights << endl;
			throw new runtime_error("decrypt failed!");
		}

		/* Load the network. */
		cout<<"loading "<<model_def<<endl;
		_net.reset(new Net<float>(model_def, deploy_content, caffe::TEST));
		cout<<"loading "<<weights<<endl;
		_net->CopyTrainedLayersFrom(weights, weight_content);
	} else {
		_net_param.reset(new caffe::NetParameter());
		SAY("load model definition: %s", model_def.c_str());
		caffe::ReadNetParamsFromTextFileOrDie(model_def, _net_param.get());
		_net_param->mutable_state()->set_phase(caffe::TEST);

		_net.reset(new Net<float>(*_net_param));
		// SAY("load weights: %s", weights.c_str());
		_net->CopyTrainedLayersFrom(weights);
	}

	_input_blob  = _net->input_blobs()[0];
    do {
        vector<int> shape = _input_blob->shape();
        shape[0] = batch_size;
        _input_blob->Reshape(shape);
        _net->Reshape();
    } while (0);
    _output_blob = _net->blob_by_name(layer_name).get();
    _feature_size = _output_blob->count() / _output_blob->shape()[0];
    // SAY("out_num = %d", _output_blob->num());
    // SAY("out_size = %d", _output_blob->count());
    // SAY("init finished, feature size = %lu", _feature_size);
}

static Mat CropImage(const Mat &image, int crop_height, int crop_width,
        int center_x, int center_y) {
    Mat result(Mat::zeros(crop_height, crop_width, image.type()));
    assert(!image.empty() && image.rows >= crop_height && image.cols >= crop_width);
    if (center_x <= 0 || center_y <= 0) {
        // TODO: if x < 0 || y < 0, why center should be set to the image center?
        center_x = image.cols / 2;
        center_y = image.rows / 2;
    }

	int left1, top1, right1, bottom1;
	int left2, top2, right2, bottom2;
	int overlop_left, overlop_top;
	int overlop_right, overlop_bottom;
	left1 = 0, top1 = 0;
	right1 = image.cols, bottom1 = image.rows;
	left2 = center_x - crop_width / 2;
	top2 = center_y - crop_height / 2;
	right2 = left2 + crop_width;
	bottom2 = top2 + crop_height;

	overlop_left = MAX(left1, left2);
	overlop_right = MIN(right1, right2);
	overlop_top = MAX(top1, top2);
	overlop_bottom = MIN(bottom1, bottom2);

	Rect roi_rect1(overlop_left, overlop_top, overlop_right - overlop_left,
		overlop_bottom - overlop_top);

	left1 = 0, top1 = 0;
	right1 = crop_width, bottom1 = crop_height;
	left2 = crop_width / 2 - center_x;
	top2 = crop_height /2 - center_y;
	right2 = left2 + image.cols;
	bottom2 = top2 + image.rows;

	overlop_left = MAX(left1, left2);
	overlop_right = MIN(right1, right2);
	overlop_top = MAX(top1, top2);
	overlop_bottom = MIN(bottom1, bottom2);

	Rect roi_rect2(overlop_left, overlop_top, overlop_right - overlop_left,
		overlop_bottom - overlop_top);
	image(roi_rect1).copyTo(result(roi_rect2)); // = image(roi_rect1);
	return result;
}

static void ReadImageToData(const Mat& img, const int height, const int width,
        const int crop_height, const int crop_width, const int inner_crop,
        const vector<float> &landmarks, float* transformed_data) {

    vector<Mat> face_patchs;
    for (int i = 0; i < landmarks.size() / 2; i++) {
        Mat crop_img, resize_img;
        int temp_crop_height, temp_crop_width;
        // crop_height and crop_width of main face patch keep to be 400
        if (landmarks[i * 2] == 0 && landmarks[i * 2 + 1] == 0) {
            temp_crop_height = 400;
            temp_crop_width = 400;
        } else if (landmarks[i * 2] == -1 && landmarks[i * 2 + 1] == -1) {
            temp_crop_height = 300;
            temp_crop_width = 300;
        } else {
            temp_crop_height = crop_height;
            temp_crop_width = crop_width;
        }
        if (i < landmarks.size() /2 - 1) {
            crop_img = CropImage(img, temp_crop_height, temp_crop_width,
                landmarks[i * 2], landmarks[i * 2 + 1]);
        } else {
            crop_img = CropImage(img, temp_crop_height, temp_crop_width,
                landmarks[i * 2], landmarks[i * 2 + 1]);
        }
        if (height > 0 && width > 0) {
            resize(crop_img, resize_img, Size(width, height));
        } else {
            resize_img = crop_img;
        }
        face_patchs.push_back(resize_img);
    }
    int num_channels = 3;

    const int crop_size = inner_crop;
    int h_off = 0;
    int w_off = 0;
    // We only do random crop when we do training.
    h_off = (height - crop_size) / 2;
    w_off = (width - crop_size) / 2;

    int top_index, data_index;
    for (int i =0; i < face_patchs.size(); i++)	{
        Mat cv_img; // = face_patchs[i].clone();
        face_patchs[i].convertTo(cv_img, CV_32FC3);
        int idx = 0;
        for (int c = 0; c < num_channels; ++c) {
            for (int h = 0; h < crop_size; ++h) {
                for (int w = 0; w < crop_size; ++w) {
                    float data_element;
                    top_index = ((i * num_channels + c) * crop_size + h) * crop_size + w;
                    data_element = cv_img.at<cv::Vec3f>(h + h_off, w + w_off)[c] / 255.f;
                    transformed_data[top_index] = data_element;
                }
            }
        }
    }
}

void CaffeBatchWrapper::ReadFacePatchImageToData(const Mat& img,
        const vector<float> & landmarks, float *transformed_data) {
  vector<float> patch_landmarks;
  float mark_x, mark_y;
  for (int i = 0; i < _patch_ids.size(); i++) {
    switch(_patch_ids[i]) {
      case main_face:
        patch_landmarks.push_back(0);
        patch_landmarks.push_back(0);
        break;
      case left_eye:
        patch_landmarks.push_back(landmarks[21 * 2 + 0]);
        patch_landmarks.push_back(landmarks[21 * 2 + 1]);
        break;
      case right_eye:
        patch_landmarks.push_back(landmarks[38 * 2 + 0]);
        patch_landmarks.push_back(landmarks[38 * 2 + 1]);
        break;
      case nose:
        patch_landmarks.push_back(landmarks[57 * 2 + 0]);
        patch_landmarks.push_back(landmarks[57 * 2 + 1]);
        break;
      case mouth:
        mark_x = (landmarks[58 * 2] + landmarks[62 * 2])/2 ;
        mark_y = (landmarks[58 * 2 + 1] + landmarks[62 * 2 + 1])/2 ;
        patch_landmarks.push_back(mark_x);
        patch_landmarks.push_back(mark_y);
        break;
      case middle_eye:
        mark_x = (landmarks[21 * 2] + landmarks[38 * 2])/2 ;
        mark_y = (landmarks[21 * 2 + 1] + landmarks[38 * 2 + 1])/2 ;
        patch_landmarks.push_back(mark_x);
        patch_landmarks.push_back(mark_y);
        break;
      case left_mouth:
        patch_landmarks.push_back(landmarks[58 * 2 + 0]);
        patch_landmarks.push_back(landmarks[58 * 2 + 1]);
        break;
      case right_mouth:
        patch_landmarks.push_back(landmarks[62 * 2 + 0]);
        patch_landmarks.push_back(landmarks[62 * 2 + 1]);
        break;
      case mini_face:
        patch_landmarks.push_back(-1);
        patch_landmarks.push_back(-1);
        break;
      case left_brow:
        patch_landmarks.push_back(landmarks[24*2]);
        patch_landmarks.push_back(landmarks[24*2+1]);
        break;
      case right_brow:
        patch_landmarks.push_back(landmarks[41*2]);
        patch_landmarks.push_back(landmarks[41*2+1]);
        break;
      case middle_brow:
        mark_x = (landmarks[24 * 2] + landmarks[41 * 2])/2 ;
        mark_y = (landmarks[24 * 2 + 1] + landmarks[41 * 2 + 1])/2 ;
        patch_landmarks.push_back(mark_x);
        patch_landmarks.push_back(mark_y);
        break;
      default:
        patch_landmarks.push_back(0);
        patch_landmarks.push_back(0);
        break;
    }
  }
  ReadImageToData(img, 128, 128, 192, 192, 108,
      patch_landmarks, transformed_data);
}

void CaffeBatchWrapper::predict(const vector<Mat> &images,
        const vector< vector<float> > &landmarks_list,
        vector< vector<float> > &output) {
	//cout << "batch_size: " << _batch_size << ", image size: " << images.size() << endl;
    assert(_batch_size >= images.size());
	assert(_batch_size == _input_blob->shape()[0]);
	assert(_batch_size == _output_blob->shape()[0]);
    int patch_count  = _input_blob->shape()[1];
    int patch_height = _input_blob->shape()[2];
    int patch_width  = _input_blob->shape()[3];
    size_t image_size = patch_count * patch_height * patch_width;
    float *data = _input_blob->mutable_cpu_data();
    for (size_t i = 0; i < images.size(); i++) {
        const Mat &img = images[i];
        const vector<float> &landmarks = landmarks_list[i];
        assert(img.rows == 600 && img.cols == 600 && img.channels() == 3);
        ReadFacePatchImageToData(img, landmarks, data + i * image_size);
    }
    float loss;
    _net->Forward(&loss);
    output.clear();
    output.resize(images.size());
    const float *out = _output_blob->cpu_data();
    for (size_t i = 0; i < images.size(); i++) {
        output[i].resize(_feature_size);
        memcpy(&output[i][0], out + i * _feature_size,
			_feature_size * sizeof(float));
    }
}

