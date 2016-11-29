#include "non_motor_vehicle_classifier.h"
#include <algorithm>

// #define SHOW_DEBUG

namespace dg {

NonMotorVehicleClassifier::NonMotorVehicleClassifier(NonMotorVehicleConfig &nonMotorVehicleConfig) {
    config = nonMotorVehicleConfig;
}

NonMotorVehicleClassifier::~NonMotorVehicleClassifier() {

}
// CaffeAttribute::Attrib
void NonMotorVehicleClassifier::BatchClassify(const vector<cv::Mat> &images, vector <vector<CaffeAttribute::Attrib>> &results) {
    Faster_rcnn detector(config.rpn_deploy_file, config.rpn_trained_file, layer_name_cls,
                         layer_name_reg, config.use_gpu, image_size, det_thresh, max_per_img, sliding_window_stride, area, ratio);

    CaffeAttribute bitri_attrib(config.attrib_table_path, config.bitri_deploy_file,
                                config.bitri_trained_file, config.bitri_layer_name,
                                360, 360, 300, 300, 256, config.use_gpu);

    CaffeAttribute upper_attrib(config.attrib_table_path, config.upper_deploy_file,
                                config.upper_trained_file, config.upper_layer_name,
                                180, 180, 150, 150, 256, config.use_gpu);

    const vector<CaffeAttribute::Attrib> attrib_table = bitri_attrib._attrib_table;

    size_t num_attribs = attrib_table.size();

    vector<bool> upper_body_table(num_attribs);
    for (size_t i = 0; i < num_attribs; i++) {
        string name = attrib_table[i].name;
        upper_body_table[i] = attrib_table[i].name.find("transportation_") == string::npos;
    }


    //int batch_size = pedestrian_attrib._batch_size;
    int batch_size = bitri_attrib._batch_size;

    // bitricycle overall attributes prediction
    vector<vector<float> > results_bitri;
    cout << "bitri attributes prediction " << endl;
    bitri_attrib.AttributePredict(images, results_bitri);

    // upper body attributes prediction
    vector<vector<float> > results_upper;
    vector< vector<Rect> > bboxes_list;
    bboxes_list.resize(images.size());
    cout << "processins batch after detected " << endl;
    process_batch_after_det(upper_attrib, images, bboxes_list, results_upper, batch_size);

    // merge bitricycle and upper body attributes predictions
    results.clear();
    for (size_t i = 0; i < results_bitri.size(); ++i) {
        std::vector<float> v;
        for (size_t j = 0; j < num_attribs; ++j) {
            if (j >= results_bitri[i].size()) {
                v.push_back(-1.00);
            } else {
                float result = results_bitri[i][j];
                // if upper body bbox has been detected
                if (results_upper[i].size()) {
                    if (!upper_body_table[i]) {
                        // update the upper body attributes predicted by upper atttrib model
                        result = results_upper[i][j]; 
                    }
                }
                v.push_back(result);
            }
        }
        results.push_back(attrib_table);
        for (size_t j = 0; j < results[results.size() - 1].size(); ++j) {
            results[results.size() - 1][j].confidence = v[j];
        }
    }
}

void NonMotorVehicleClassifier::process_batch_after_det(CaffeAttribute &upper_attrib, 
    const vector<Mat> &images_ori, 
    const vector< vector<Rect> > &bboxes_list, 
    vector<vector<float> > &results, size_t batch_size) {

    vector<Mat> images;
    vector<bool> flags(images_ori.size(), 0);
    for (size_t i = 0; i < images_ori.size(); i++) {
        Mat image = images_ori[i];
        if (bboxes_list[i].size()) {
            // Use the top 1 bbox as the upper body roi
            const Rect &bbox = bboxes_list[i][0];
            if (bbox.width > 10 && bbox.height > 10 && (bbox.x + bbox.width) <= image.cols && (bbox.y + bbox.height) <= image.rows) {
                flags[i] = true;
                image = image(bbox);
            }
        }
        images.push_back(image);
    }
    // fill dummy images for batch (only need to be applied for the upper body attrib recognizer)
    while (images.size() < batch_size) {
        images.push_back(images[0]);
    }
    upper_attrib.AttributePredict(images, results);

    // remove dummy images 
    results.resize(images.size());  
    // remove meaningless results with inputs that have not detected any upper body
    for (size_t i = 0; i < images.size(); i++) {
        if (!flags[i]) {
            results[i].clear();
        }
    }
}

bool mycmp(struct Bbox b1, struct Bbox b2)
{
    return b1.confidence > b2.confidence;
}

Faster_rcnn::Faster_rcnn(const string& model_file,
                         const string& trained_file,
                         const string& layer_name_cls,
                         const string& layer_name_reg,
                         const bool use_GPU,
                         const Size &image_size,
                         const float conf_thres,
                         const int max_per_img,
                         const int sliding_window_stride,
                         const vector<float> &area,
                         const vector<float> &ratio)
{
    image_size_ = image_size;
    if (use_GPU)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(0);
        useGPU_ = true;
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
        useGPU_ = false;
    }


    /* Load the network. */
//    cout<<"loading "<<model_file<<endl;
    net_.reset(new Net<float>(model_file, TEST));
//    cout<<"loading "<<trained_file<<endl;
    net_->CopyTrainedLayersFrom(trained_file);
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    /* Set batchsize */
    batch_size_ = input_layer->num();

    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    pixel_means_.push_back(102.9801);
    pixel_means_.push_back(115.9465);
    pixel_means_.push_back(122.7717);
    conf_thres_  = conf_thres;
    max_per_img_ = max_per_img;
    layer_name_cls_ = layer_name_cls;
    layer_name_reg_ = layer_name_reg;
    sliding_window_stride_ = sliding_window_stride;
    area_  = area;
    ratio_ = ratio;

    do
    {
        vector<int> shape = {batch_size_, 3, image_size_.height, image_size_.width};
        input_layer->Reshape(shape);
        net_->Reshape();
    }
    while (0);
}

// preprocess, add black edge to make batch processing aplicable for inputs with different image sizes
void Faster_rcnn::edge_complete(const vector<cv::Mat> &imgs, vector<cv::Mat> &imgs_new, const Size &image_size)
{
    imgs_new.resize(0);
    for(size_t i = 0; i < imgs.size(); i++)
    {
        Mat mask = Mat::zeros(image_size,CV_8UC1);
        Rect img_rect(0, 0, 0, 0);
        Mat img = imgs[i];
        float resize_ratio = 1.0f;
        Size img_size_new;
        float ratio_rows = image_size.height / float(img.rows) + 1e-6;
        float ratio_cols = image_size.width  / float(img.cols) + 1e-6;

        
        if (ratio_rows > ratio_cols)
        {
            resize_ratio = ratio_cols;
        } 
        else 
        {
            resize_ratio = ratio_rows;
        }
        img_size_new  = Size(img.cols * resize_ratio, img.rows * resize_ratio);
        resize(img, img, img_size_new);
        img_rect.width  = img_size_new.width;
        img_rect.height = img_size_new.height;
        mask(img_rect) = img;
        imgs_new.push_back(mask);
    } 
}

// predict single frame forward function
void Faster_rcnn::forward(const vector< cv::Mat > &imgs, vector<Blob<float>* > &outputs)
{

    Blob<float>* input_layer = net_->input_blobs()[0];
    assert(static_cast<int>(imgs.size()) <= batch_size_ && imgs.size());

    if (static_cast<int>(imgs.size()) != batch_size_)
    {
        vector<int> shape = {static_cast<int>(imgs.size()), 3, image_size_.height, image_size_.width};
        input_layer->Reshape(shape);
        net_->Reshape();
    }

    vector<cv::Mat> imgs_new;
    edge_complete(imgs, imgs_new,  image_size_);
    for(size_t i = 0; i < imgs_new.size(); i++)
    {
        Mat sample;
        Mat img = imgs_new[i];
        assert(img.rows == image_size_.height && img.cols == image_size_.width);
        if (img.channels() == 3 && num_channels_ == 1)
            cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
            cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
            cvtColor(img, sample, CV_RGBA2BGR);
        else if (img.channels() == 1 && num_channels_ == 3)
            cvtColor(img, sample, CV_GRAY2BGR);
        else
            sample = img;

        float* input_data = input_layer->mutable_cpu_data();
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
                        float(sample.at<uchar>(row, col * 3 + k)) - pixel_means_[k];
                }
            }
        }
    }
    net_->ForwardPrefilled();
    if(useGPU_)
    {
        cudaDeviceSynchronize();
    }

    outputs.resize(0);
    auto output_cls = net_->blob_by_name(layer_name_cls_).get();
    auto output_reg = net_->blob_by_name(layer_name_reg_).get();
    outputs.push_back(output_cls);
    outputs.push_back(output_reg);
}

void Faster_rcnn::nms_zz(vector<struct Bbox>& p, float threshold, int num_thresh = 2)
{
    sort(p.begin(), p.end(), mycmp);
    int num_dup[p.size()];
    memset(num_dup, 0, sizeof(num_dup)); 
    for(size_t i = 0; i < p.size(); i++)
    {
        if(p[i].deleted) continue;
        for(size_t j = i+1; j < p.size(); j++)
        {

            if(!p[j].deleted)
            {
                cv::Rect intersect = p[i].rect & p[j].rect;
                float iou = intersect.area() * 1.0f / (p[i].rect.area() + p[j].rect.area() - intersect.area()); //p[j].rect.area(); 
                if (iou > threshold)
                {
                    p[j].deleted = true;
                    num_dup[i] += 1;
                }
            }
        }
    }

    for(size_t i = 0; i < p.size(); i++)
    {
        if (num_dup[i] < num_thresh) {
            p[i].deleted = true;
        }
    }
}

void Faster_rcnn::nms(vector<struct Bbox>& p, float threshold)
{
    sort(p.begin(), p.end(), mycmp);
    for(size_t i = 0; i < p.size(); i++)
    {
        if(p[i].deleted) continue;
        for(size_t j = i+1; j < p.size(); j++)
        {

            if(!p[j].deleted)
            {
                cv::Rect intersect = p[i].rect & p[j].rect;
                float iou = intersect.area() * 1.0f / (p[i].rect.area() + p[j].rect.area() - intersect.area()); //p[j].rect.area(); 
                if (iou > threshold)
                {
                    p[j].deleted = true;
                }
            }
        }
    }
}

void Faster_rcnn::get_detection(vector<Blob<float>* >& outputs, vector< vector<struct Bbox> > &final_vbbox)
{
    Blob<float>*  cls   = outputs[0];
    Blob<float>*  reg   = outputs[1];

    final_vbbox.resize(0);
    final_vbbox.resize(cls->num());
    int scale_num = area_.size() * ratio_.size();

/**
#ifdef SHOW_DEBUG
    cout << "[debug num, channels, height, width] " << cls->num() << " " << cls->channels() << " " << cls->height() << " " << cls->width() << endl;
    cout << "[scale_num]" << scale_num << endl;
#endif // SHOW_DEBUG
**/
    assert(cls->channels() == scale_num * 2);
    assert(reg->channels() == scale_num * 4);

    //cls->Reshape(cls->num() * scale_num, cls->channels() / scale_num, cls->height(), cls->width());
    //reg->Reshape(reg->num() * scale_num, reg->channels() / scale_num, reg->height(), reg->width());

    //assert(cls->channels() == 2);
    //assert(reg->channels() == 4);
    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());
/**
#ifdef SHOW_DEBUG
    cout << "[debug stride, h,w] " <<  sliding_window_stride_ << " " << cls->height()  << " " << cls->width()<< endl;
#endif // SHOW_DEBUG
**/
    vector<struct Bbox> vbbox;
    const float* cls_cpu = cls->cpu_data();
    const float* reg_cpu = reg->cpu_data();

    vector<float> gt_ww, gt_hh;
    gt_ww.resize(scale_num);
    gt_hh.resize(scale_num);

    for(size_t i = 0; i < area_.size(); i++)
    {
        for(size_t j = 0; j < ratio_.size(); j++)
        {
            int index = i * ratio_.size() + j;
            gt_ww[index] = sqrt(area_[i] * ratio_[j]);
            gt_hh[index] = gt_ww[index] / ratio_[j];
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
                        x0 -= min_01;
                        x1 -= min_01;
                        confidence = exp(x1) / (exp(x1) + exp(x0));
                    }
                    if( confidence > conf_thres_)
                    {
                        for(int j = 0; j < 4; j++)
                        {
                            rect[j] = reg_cpu[reg_index + j * skip];
                        }

                        float shift_x = w * sliding_window_stride_ + sliding_window_stride_ / 2.f - 1;
                        float shift_y = h * sliding_window_stride_ + sliding_window_stride_ / 2.f - 1;
                        rect[2] = exp(rect[2]) * gt_ww[scale_idx];
                        rect[3] = exp(rect[3]) * gt_hh[scale_idx];
                        rect[0] = rect[0] * gt_ww[scale_idx] - rect[2] / 2.f + shift_x;
                        rect[1] = rect[1] * gt_hh[scale_idx] - rect[3] / 2.f + shift_y;

                        struct Bbox bbox;
                        bbox.confidence = confidence;
                        bbox.rect = Rect(rect[0], rect[1], rect[2], rect[3]);
                        bbox.rect &= Rect(0, 0, image_size_.width, image_size_.height);
                        bbox.deleted = false;
                        vbbox.push_back(bbox);
                    }

                    cls_index += 1;
                    reg_index += 1;
                }
            }
            cls_index += skip;
            reg_index += 3 * skip;
        }
        nms_zz(vbbox, 0.3);
        // nms(vbbox, 0.3);
        //nms(vbbox, 0.2);
        for(size_t i = 0; i < vbbox.size(); i++)
        {
            if(!vbbox[i].deleted)
            {
                final_vbbox[img_idx].push_back(vbbox[i]);
            }
        }
    }
}

CaffeAttribute::CaffeAttribute(const string& attrib_table_path,
        const string& model_file,
        const string& trained_file,
        const string& layer_name,
        const int height,
        const int width,
        const int crop_height,
        const int crop_width,
        const int pixel_scale,
        const bool use_GPU) : _height(height), _width(width),
        _crop_height(crop_height), _crop_width(crop_width), _pixel_scale(pixel_scale),
        _pixel_means{104, 117, 123} {

    /* Load the network. */
//    cout<<"loading "<< model_file <<endl;
    _net.reset(new Net<float>(model_file, TEST));
 //   cout<<"loading "<< trained_file <<endl;
    _net->CopyTrainedLayersFrom(trained_file);
    _layer_name = layer_name;

    Blob<float>* input_blob = _net->input_blobs()[0];
    _batch_size   = input_blob->num();
    _num_channels = input_blob->channels();
    CHECK(_num_channels == 3) << "Input layer should have 3 channels.";
    input_blob->Reshape(_batch_size, _num_channels,
        _crop_height, _crop_width);

    if (use_GPU) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(0);
        _useGPU = true;
    }
    else {
        Caffe::set_mode(Caffe::CPU);
        _useGPU = false;
    }

    // load attrib names
    load_names(attrib_table_path, _attrib_table);

    //calculate crop rectangle coordinates
    int offset_h = _height - _crop_height;
    int offset_w = _width  - _crop_width;
    offset_h = offset_h / 2;
    offset_w = offset_w / 2;
    _crop_rect = Rect(offset_w, offset_h, _crop_width, _crop_height);
}

void CaffeAttribute::load_names(const string &name_list, vector<CaffeAttribute::Attrib> &attribs) {
    ifstream fp(name_list);
    attribs.resize(0);
    while (!fp.eof()) {
        string name = "", thresh_low_str = "", thresh_high_str = "";
        string idx, confidence, mappingId, categoryId;
        CaffeAttribute::Attrib attrib;
        fp >> name >> idx >> categoryId >> mappingId;
        fp >> thresh_low_str;
        fp >> thresh_high_str;
        if (name == "" || thresh_low_str == "" || thresh_high_str == "")
            continue;
        if (idx == "" || categoryId == "" || mappingId == "")
            continue;
        attrib.name   = name;
        attrib.thresh_low = atof(thresh_low_str.c_str());
        attrib.thresh_high = atof(thresh_high_str.c_str());
        attrib.idx = atoi(idx.c_str());
        attrib.categoryId = atoi(categoryId.c_str());
        attrib.mappingId = atoi(mappingId.c_str());
        attribs.push_back(attrib);
    }
}

void CaffeAttribute::BatchAttributePredict(const vector<Mat> &imgs, vector<vector<float> > &results) {
    Blob<float>* input_blob = _net->input_blobs()[0];
    int num_imgs = static_cast<int>(imgs.size());
    assert(num_imgs <= _batch_size);
    vector<int> shape = {num_imgs, 3, _crop_height, _crop_width};
    input_blob->Reshape(shape); 
    _net->Reshape();
    float* input_data = input_blob->mutable_cpu_data();
    int cnt = 0;

    for (size_t i = 0; i < imgs.size(); i++) {
        Mat sample, img = imgs[i];
        if (img.channels() == 4 && _num_channels == 3)
            cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1 && _num_channels == 3)
            cvtColor(img, sample, CV_GRAY2BGR);
        else
            sample = img;

        if((sample.rows != _height) || (sample.cols != _width)) {
            resize(sample, sample, Size(_width, _height));
            sample(_crop_rect).copyTo(sample);  
        } else {
            sample(_crop_rect).copyTo(sample);  
        }

        for(int k = 0; k < sample.channels(); k++) {
            for(int row = 0; row < sample.rows; row++) {
                for(int col = 0; col < sample.cols; col++) {
                    input_data[cnt] = (float(sample.at<uchar>(row, col * 3 + k)) - _pixel_means[k]) / _pixel_scale;
                    cnt++;
                }
            }
        }
    }

    _net->ForwardPrefilled();
    if(_useGPU)
    {
        cudaDeviceSynchronize();
    }

    auto output_blob = _net->blob_by_name(_layer_name);
    const float *output_data = output_blob->cpu_data();
    const int feature_len = output_blob->channels();
    assert(feature_len <= static_cast<int>(_attrib_table.size())); 
    //assert(feature_len == static_cast<int>(_attrib_table.size())); 
    //cerr << "output_blob:\t" << output_blob->count() << endl;
    results.resize(imgs.size());
    for (size_t i = 0; i < imgs.size(); i++) {
        const float *data = output_data + i * feature_len; 
        vector<float> &feature = results[i];
        feature.resize(feature_len);
        for (int idx = 0; idx < feature_len; ++idx) {
            feature[idx] = data[idx];
        }   
    }
}

void CaffeAttribute::AttributePredict(const vector<Mat> &imgs, vector<vector<float> > &results) {

    results.clear();
    vector<Mat> toPredict;
    for (auto i : imgs) {
        toPredict.push_back(i);
        if (toPredict.size() == _batch_size) {
            vector<vector<float>> predict_result;
            BatchAttributePredict(toPredict, predict_result);
            results.insert(results.end(), predict_result.begin(), predict_result.end());
            toPredict.clear();
        }
    }
    if (toPredict.size() > 0) {
        vector<vector<float>> predict_result;
        BatchAttributePredict(toPredict, predict_result);
        results.insert(results.end(), predict_result.begin(), predict_result.end());
    }
}

} /* namespace dg */