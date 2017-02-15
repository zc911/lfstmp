#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "detector/PyramidDenseBox.hpp"
//#include "pryd_util.hpp"
//#include "FaUtil.h"
#include <cstring>
#include<sys/time.h>

namespace db {
using namespace std;
using namespace cv;

    template <typename Dtype>
const bool nms( vector< RotateBBox<Dtype> > candidates, 
        vector< RotateBBox<Dtype> >& res, 
        const Dtype overlap, const int top_N, const bool addScore)
{
    res.clear();
    vector<bool> is_candidate_selected;
    is_candidate_selected = nms( candidates, overlap, top_N, addScore);
    for ( int i = 0 ; i < is_candidate_selected.size() ; ++i )
    {
        if( is_candidate_selected[i] )
            res.push_back(candidates[i]);
    }
    return(true);
}
template const bool nms( vector< RotateBBox<float> > candidates, 
        vector< RotateBBox<float> >& res, 
        const float overlap, const int top_N, const bool addScore);
template const bool nms( vector< RotateBBox<double> > candidates, 
        vector< RotateBBox<double> >& res, 
        const double overlap, const int top_N, const bool addScore);

bool PyramidDenseBox::setDetSize(const int imgWidth, const int imgHeight, const int template_size, 
        const int minDetFaceSize, const int maxDetFaceSize, 
        const int minImgSize, const int maxImgSize, 
        const float minScaleFaceToImg, const float maxScaleFaceToImg, 
        float& scale_start, float& scale_end){
    assert( template_size > 0);
    assert( minScaleFaceToImg > 0 && minScaleFaceToImg <= maxScaleFaceToImg );

    if( imgWidth <=0 || imgHeight<= 0 )
    {
        //LOG(INFO) << "imgWidth=" << imgWidth <<", imgHeight=" << imgHeight;
        return(false);
    }

    float size = max( imgWidth, imgHeight ) ;
    scale_start = (float)template_size / ( maxScaleFaceToImg*size );
    scale_end   = (float)template_size / ( minScaleFaceToImg*size );

    if( minDetFaceSize > 0 )
    {
        scale_end = min( scale_end, (float)template_size / minDetFaceSize );
        scale_start = min( scale_start, scale_end);
    }
    if( maxDetFaceSize > 0 )
    {
        scale_start = max( scale_start, (float)template_size / maxDetFaceSize );
        scale_end = max( scale_end, scale_start);
    }
    if( minImgSize > 0)
        scale_start = max( scale_start, ((float)minImgSize/max(imgWidth, imgHeight) ) );
    if( maxImgSize > 0)
        scale_end   = min( scale_end,   ((float)maxImgSize/min(imgWidth, imgHeight) ) );

    assert( scale_end >= scale_start );
    return(true);
}

bool PyramidDenseBox::constructPyramidImgs(const Mat & img, vector<Mat>& pyramidImgs)
{
    float scale_start = 0.1, scale_end = 1.0;
    int imgWidth  = img.cols,
        imgHeight = img.rows;
    setDetSize(imgWidth, imgHeight, templateSize_, minDetFaceSize_, maxDetFaceSize_, 
            minImgSize_, maxImgSize_, minScaleFaceToImg_, maxScaleFaceToImg_, 
            scale_start, scale_end);

    //std::cout << "[scale_end_start: ] " << scale_end << " " << scale_start << std::endl;

    pyramidImgs.clear();
    Mat tempImg;
    float scale_tmp = scale_end;
    Size dst_size( int(img.cols*scale_tmp),int(img.rows*scale_tmp) );
    if(!dst_size.area())
        return false;
    resize( img, tempImg,dst_size);
    pyramidImgs.push_back(tempImg);
    //std::cout << "[debug pyramids] " << dst_size.width << " " << dst_size.height << std::endl;
    int idx = 0;
    while( scale_tmp >= scale_start )
    {
        scale_tmp /= stepScale_;
        Size dst_size( int(img.cols*scale_tmp),int(img.rows*scale_tmp) );
        //std::cout << "[debug pyramids] " << dst_size.width << " " << dst_size.height << std::endl;
        if(!dst_size.area())
            break;
        resize( pyramidImgs[idx], tempImg, dst_size);
        pyramidImgs.push_back(tempImg);
        idx++;


    }

    return true;

}

    template <typename Dtype> 
void PyramidDenseBox::setImgDenseBox(Dtype* pblob, Mat& img, Scalar img_mean, float mvnPower, float mvnScale, float mvnShift)
{
    //timer timer1;
    //timer1.tic();
    int channels = img.channels();
    int rows = img.rows;
    int cols = img.cols;
    assert(mvnPower==1.0f);
    Mat img_32f;
    img.convertTo(img_32f, CV_32FC3, mvnScale,mvnShift);
    Scalar img_mean_32f(img_mean[0] * mvnScale, img_mean[1] * mvnScale, img_mean[2] * mvnScale);
    subtract(img_32f,img_mean_32f,img_32f);
    //cerr<<"normalize img cost:"<<timer1.toc()*1000<<" ms"<<endl;
    //timer1.tic();
    vector<Mat> bgrchannels;
    split(img_32f,bgrchannels);
    int img_size = rows * cols;
    for (int c = 0; c < channels; ++c) 
    {
        memcpy(pblob +  c * img_size, bgrchannels[c].ptr(0),img_size*sizeof(float));
        /*
        for (int h = 0; h < rows; ++h) 
        {
            float* pixel = img_32f.ptr<float>(h)+c;
            for (int w = 0; w < cols; ++w) 
            {
                *pblob++ = *pixel;//(static_cast<float>(img.at<cv::Vec3b>(h, w)[c])-img_mean[c])*mvnScale+mvnShift;
                pixel += channels;
                // *pblob++ = (static_cast<float>(img.at<cv::Vec3b>(h, w)[c])-img_mean[c])*mvnScale+mvnShift;
            }
        }*/
    }
    //cerr<<"set img cost:"<<timer1.toc()*1000<<" ms"<<endl;
}

void PyramidDenseBox::predictDenseBox(caffe::shared_ptr<caffe::Net<float> > caffe_net,
	       				const Mat& org_img, 
					const vector<Mat>& pyramid_imgs, 
					vector< RotateBBox<float> >& faces) {
	// find out the max image in pyramid
	int pyramid_num = pyramid_imgs.size();
	vector<float> pyramid_scale(pyramid_num, 1.0);
	int max_pyr_img = -1;
	float max_scale = 0;
	for(int i = 0; i < pyramid_num; ++i) {
		pyramid_scale[i] = static_cast<float>(pyramid_imgs[i].rows) / org_img.rows;
		if(max_scale > pyramid_scale[i]) {
			max_pyr_img = i;
			max_scale = pyramid_scale[i];
		}
	}

    int desHeight = (pyramid_imgs[max_pyr_img].rows+pad_h_*2+max_stride_-1)/max_stride_*max_stride_;
    int desWidth = (pyramid_imgs[max_pyr_img].cols+pad_w_*2+max_stride_-1)/max_stride_*max_stride_;
    int desChannel = pyramid_imgs[max_pyr_img].channels();


    vector<int> blobshape;
    blobshape.push_back(pyramid_num);//num
    blobshape.push_back(desChannel);//channels
    blobshape.push_back(desHeight);//height
    blobshape.push_back(desWidth);//width

    caffe::Blob<float> *input_blob = caffe_net->input_blobs()[0];
    input_blob->Reshape(blobshape);
    caffe_net->Reshape();
    float *blobData = input_blob->mutable_cpu_data();
	
	// float* blobData = new float[pyramid_num * desChannel * desHeight * desWidth];
    for(int i = 0; i < pyramid_num; ++i) {
    	Mat srcImg(desHeight, desWidth, pyramid_imgs[i].type(), Scalar(0,0,0));
    	Rect roi = Rect(pad_w_, pad_h_, pyramid_imgs[i].cols, pyramid_imgs[i].rows);
    	pyramid_imgs[i].copyTo(srcImg(roi));
    	float* curr_data = blobData + i * desChannel * desHeight * desWidth;
    	setImgDenseBox(curr_data, srcImg, Scalar(mean_b_, mean_g_, mean_r_), mvnPower_, mvnScale_, mvnShift_);
    }
    //cerr<<"setImgDenseBox img cost:"<<timer1.toc()*1000<<" ms"<<endl;

    //timer1.tic();

    //std::cout << "densebox input" << 1 << " " << srcImg.channels() << " " << srcImg.rows << " " << srcImg.cols << std::endl;
    
    caffe_net->Forward(NULL);
    cudaDeviceSynchronize();
    //cerr<<"preprocess image cost:"<<timer1.toc()*1000<<" ms"<<endl;

    //timer1.tic();
    caffe::Blob<float> *output_blob = caffe_net->blob_by_name("res_all").get();
    vector<int> outblob_shape = output_blob->shape();

    //cerr<<"caffe predict cost:"<<timer1.toc()*1000<<" ms"<<endl;
    //timer1.tic();

    int num = outblob_shape[0];//outBlob[0]->num();
    //CHECK_EQ(num, 1);

    const int map_height = outblob_shape[2];//outBlob[0]->height();
    const int map_width = outblob_shape[3];//outBlob[0]->width();
    const int map_size = map_height * map_width;
    const int scale_num = outblob_shape[1]/channel_per_scale_;//outBlob[0]->channels()/channel_per_scale_;

    const float pad_h_map = pad_h_/heat_map_a_;
    const float pad_w_map = pad_w_/heat_map_a_;

    //CHECK_EQ(outblob_shape[1]%channel_per_scale_,0);
    //CHECK_EQ((outblob_shape[1]/scale_num)/channel_per_scale_, class_num_);
    //CHECK_EQ((outblob_shape[1]/scale_num)%channel_per_scale_, 0);

    /**
     * Get bbox from botom[0]
     */
	faces.clear();
    // const float* bottom_data = outputblobs[0]->cpu_data(); //outBlob[0]->mutable_cpu_data();
    const float* bottom_data = output_blob->cpu_data(); //outBlob[0]->mutable_cpu_data();
    for( int i = 0 ; i < num ; ++ i ) {
		vector< vector< RotateBBox<float> > > all_candidates(class_num_);

        for( int scale_id = 0 ; scale_id < scale_num ; ++scale_id ) {
            for( int class_id = 0 ; class_id < class_num_; ++class_id) {
                int score_channel = ((i*scale_num + scale_id)*class_num_ + class_id )* channel_per_scale_;;
                int offset_channel = score_channel + 1;

                const float* curScores = bottom_data + ((i * outblob_shape[1] + score_channel) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,score_channel  , 0,0);
                const float* dx1 =    bottom_data + ((i * outblob_shape[1] + offset_channel+0) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+0,0,0);
                const float* dy1 =    bottom_data + ((i * outblob_shape[1] + offset_channel+1) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+1,0,0);
                const float* dx2 =    bottom_data + ((i * outblob_shape[1] + offset_channel+2) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+2,0,0);
                const float* dy2 =    bottom_data + ((i * outblob_shape[1] + offset_channel+3) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+3,0,0);

                for(int off = 0; off< map_size; ++off) {
                    if(curScores[off] < nms_threshold_)
                        continue;
                    int h = off / map_width;
                    int w = off % map_width;
                    if(h >=  pad_h_map && h < map_height - pad_h_map && w >= pad_w_map  && w < map_width - pad_w_map) {
                        float cur_h = h * heat_map_a_ + heat_map_b_ - pad_h_;
                        float cur_w = w * heat_map_a_ + heat_map_b_ - pad_w_;	

                        float left_top_x = cur_w - dx1[off]*heat_map_a_;
                        float left_top_y = cur_h - dy1[off]*heat_map_a_;
                        float right_bottom_x = cur_w - dx2[off]*heat_map_a_;
                        float right_bottom_y = cur_h - dy2[off]*heat_map_a_;

                        RotateBBox<float> candidate( left_top_x, left_top_y, right_bottom_x, right_bottom_y, curScores[off]);
                        all_candidates[class_id].push_back(candidate);
                    }
                } // for(int off = 0; off< map_size; ++off) 

            } // for( int class_id = 0 ; class_id < class_num_; ++class_id) 
        } // for( int scale_id = 0 ; scale_id < scale_num ; ++scale_id ) 
		
		//nms
		vector< vector< RotateBBox<float> > > nms_output_candidates(class_num_);
		for( int class_id = 0 ; class_id < class_num_ ; ++class_id ) {
			vector< RotateBBox<float> >& cur_candidates = all_candidates[class_id];
			vector< RotateBBox<float> >& cur_output     = nms_output_candidates[class_id];
			nms( cur_candidates, cur_output, (float)nms_overlap_ratio_, (int)nms_top_n_,false);

			for(size_t rbbox_idx = 0; rbbox_idx < cur_output.size(); ++rbbox_idx) {
				//back to the orignal image
				float ratio_x = float(org_img.cols) / pyramid_imgs[i].cols;
				float ratio_y = float(org_img.rows) / pyramid_imgs[i].rows;

				RotateBBox<float>& curr_rbbox = cur_output[rbbox_idx];
				curr_rbbox.lt_x *= ratio_x;
				curr_rbbox.lt_y *= ratio_y;
				curr_rbbox.rt_x *= ratio_x;
				curr_rbbox.rt_y *= ratio_y;
				curr_rbbox.rb_x *= ratio_x;
				curr_rbbox.rb_y *= ratio_y;                 			 
				curr_rbbox.lb_x *= ratio_x;
				curr_rbbox.lb_y *= ratio_y;                 			 
			}
		} // for( int class_id = 0 ; class_id < class_num_ ; ++class_id ) 
		faces.insert(faces.end(), nms_output_candidates[0].begin(), nms_output_candidates[0].end());
    } // for( int i = 0 ; i < num ; ++ i ) 

}
void PyramidDenseBox::DBoxMatToBlob(const Mat& org_img, Size extend_size, float* blob_data) {
	Mat srcImg(extend_size, org_img.type(), Scalar(0,0,0));
	Rect roi(pad_w_, pad_h_, org_img.cols, org_img.rows);
	org_img.copyTo(srcImg(roi));
	setImgDenseBox(blob_data, srcImg, Scalar(mean_b_, mean_g_, mean_r_), mvnPower_, mvnScale_, mvnShift_);
}

void PyramidDenseBox::predictDenseBox( caffe::shared_ptr<caffe::Net<float> > caffe_net, 
					const vector<Mat>& imgs, 
					vector<vector<RotateBBox<float> > >& faces) {
	faces.clear();
	for(size_t i = 0; i < imgs.size(); ++i) {
		if(imgs[i].size() != imgs[0].size() || imgs[i].channels() != imgs[0].channels()) {
			cout << "images size not consistant!" << endl;
			return;
		}
	}	
	faces.resize(imgs.size());
	
	int desHeight = (imgs[0].rows+pad_h_*2+max_stride_-1)/max_stride_*max_stride_;
	int desWidth = (imgs[0].cols+pad_w_*2+max_stride_-1)/max_stride_*max_stride_;
	int desChannel = imgs[0].channels();

        vector<int> blobshape;
        blobshape.push_back(imgs.size());//num
        blobshape.push_back(desChannel);//channels
        blobshape.push_back(desHeight);//height
        blobshape.push_back(desWidth);//width

        caffe::Blob<float> *input_blob = caffe_net->input_blobs()[0];
        input_blob->Reshape(blobshape);
        caffe_net->Reshape();
        float *blobData = input_blob->mutable_cpu_data();

	// for(size_t i = 0; i < imgs.size(); ++i) {
	// 	Mat srcImg(desHeight, desWidth, CV_8UC3, Scalar(0,0,0));
	// 	Rect roi(pad_w_, pad_h_, imgs[i].cols, imgs[i].rows);
	// 	imgs[i].copyTo(srcImg(roi));
	// 	float* curr_data = blobData + i * desChannel * desHeight * desWidth;
	// 	setImgDenseBox(curr_data, srcImg, Scalar(mean_b_, mean_g_, mean_r_), mvnPower_, mvnScale_, mvnShift_);
	// }
#pragma omp parallel for
	for(size_t i = 0; i < imgs.size(); ++i) {
		float* curr_data = blobData + i * desChannel * desHeight * desWidth;
		DBoxMatToBlob( imgs[i], Size(desWidth, desHeight), curr_data);
	}


	//std::cout << "densebox input" << imgs.size() << " " << srcImg.channels() << " " << srcImg.rows << " " << srcImg.cols << std::endl;
    	caffe_net->Forward(NULL);
	cudaDeviceSynchronize();
	//cerr<<"preprocess image cost:"<<timer1.toc()*1000<<" ms"<<endl;

    	caffe::Blob<float> *output_blob = caffe_net->blob_by_name("res_all").get();
    	vector<int> outblob_shape = output_blob->shape();
	//timer1.tic();

	int num = outblob_shape[0];//outBlob[0]->num();
	//CHECK_EQ(num, 1);

	const int map_height = outblob_shape[2];//outBlob[0]->height();
	const int map_width = outblob_shape[3];//outBlob[0]->width();
	const int map_size = map_height * map_width;
	const int scale_num = outblob_shape[1]/channel_per_scale_;//outBlob[0]->channels()/channel_per_scale_;

	const float pad_h_map = pad_h_/heat_map_a_;
	const float pad_w_map = pad_w_/heat_map_a_;

	const float* bottom_data = output_blob->cpu_data(); //outBlob[0]->mutable_cpu_data();
	for(int i = 0; i < num; ++i) {
		vector<int> one_img_output_shape(outblob_shape);
		one_img_output_shape[0] = 1;
		dataToBBox(bottom_data + i*outblob_shape[1]*map_size, one_img_output_shape, faces[i]);
	}

}

// convert bboxes from a image output
void PyramidDenseBox::dataToBBox(const float* output_data, const vector<int>& data_shape, vector<RotateBBox<float> >& faces) {
	// CHECK_EQ(data_shape[0] == 1);
	const int map_height = data_shape[2];//outBlob[0]->height();
	const int map_width = data_shape[3];//outBlob[0]->width();
	const int map_size = map_height * map_width;
	const int scale_num = data_shape[1]/channel_per_scale_;//outBlob[0]->channels()/channel_per_scale_;

	const float pad_h_map = pad_h_/heat_map_a_;
	const float pad_w_map = pad_w_/heat_map_a_;

	vector<vector<RotateBBox<float> > > all_candidates(class_num_);
	for(int scale_id = 0; scale_id < scale_num; ++scale_id) {
		for(int class_id = 0; class_id < class_num_; ++class_id) {
			int score_channel = (scale_id*class_num_ + class_id)*channel_per_scale_;
			int offset_channel = score_channel + 1;

			const float* curScores = output_data + (( score_channel) * data_shape[2] + 0) * data_shape[3] + 0;//outBlob[0]->offset(i,score_channel  , 0,0);
			const float* dx1 =    output_data + (( offset_channel+0) * data_shape[2] + 0) * data_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+0,0,0);
			const float* dy1 =    output_data + (( offset_channel+1) * data_shape[2] + 0) * data_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+1,0,0);
			const float* dx2 =    output_data + (( offset_channel+2) * data_shape[2] + 0) * data_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+2,0,0);
			const float* dy2 =    output_data + (( offset_channel+3) * data_shape[2] + 0) * data_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+3,0,0);
			for(int off = 0; off< map_size; ++off) {
			    if(curScores[off] < nms_threshold_) continue;
			    int h = off / map_width;
			    int w = off % map_width;
			    if(h >=  pad_h_map && h < map_height - pad_h_map && w >= pad_w_map  && w < map_width - pad_w_map) {
				float cur_h = h * heat_map_a_ + heat_map_b_ - pad_h_;
				float cur_w = w * heat_map_a_ + heat_map_b_ - pad_w_;	

				float left_top_x = cur_w - dx1[off]*heat_map_a_;
				float left_top_y = cur_h - dy1[off]*heat_map_a_;
				float right_bottom_x = cur_w - dx2[off]*heat_map_a_;
				float right_bottom_y = cur_h - dy2[off]*heat_map_a_;

				RotateBBox<float> candidate( left_top_x, left_top_y, right_bottom_x, right_bottom_y, curScores[off]);
				all_candidates[class_id].push_back(candidate);
			    }
			}
		}
	}
	//nms
	vector< vector< RotateBBox<float> > > output_candidates(class_num_);
	for( int class_id = 0 ; class_id < class_num_ ; ++class_id ) {
		vector< RotateBBox<float> >& cur_candidates = all_candidates[class_id];
		vector< RotateBBox<float> >& cur_output     = output_candidates[class_id];
		nms( cur_candidates, cur_output, (float)nms_overlap_ratio_, (int)nms_top_n_,false);
	}

	faces = output_candidates[0];
}

void PyramidDenseBox::predictDenseBox(caffe::shared_ptr<caffe::Net<float> > caffe_net, 
					Mat& img, 
					vector< RotateBBox<float> >& faces)
{
    //timer timer1;
    //timer1.tic();
    // preprocess image
    int desHeight = (img.rows+pad_h_*2+max_stride_-1)/max_stride_*max_stride_;
    int desWidth = (img.cols+pad_w_*2+max_stride_-1)/max_stride_*max_stride_;
    Mat srcImg(desHeight, desWidth, CV_8UC3, Scalar(0,0,0));
    Rect roi = Rect(pad_w_, pad_h_, img.cols, img.rows);
    //img.convertTo(srcImg(roi), srcImg.type(), 1, 0);
    img.copyTo(srcImg(roi));
    //cerr<<"convert img cost:"<<timer1.toc()*1000<<" ms"<<endl;
    //timer1.tic();
    
    vector<int> blobshape;
    blobshape.push_back(1);//num
    blobshape.push_back(srcImg.channels());//channels
    blobshape.push_back(srcImg.rows);//height
    blobshape.push_back(srcImg.cols);//width
    
    caffe::Blob<float> *input_blob = caffe_net->input_blobs()[0];
    input_blob->Reshape(blobshape);
    caffe_net->Reshape();
    float *blobData = input_blob->mutable_cpu_data();
    //setImgDenseBox(caffe_net, srcImg, Scalar(mean_b_, mean_g_, mean_r_), mvnPower_, mvnScale_, mvnShift_);
    setImgDenseBox(blobData, srcImg, Scalar(mean_b_, mean_g_, mean_r_), mvnPower_, mvnScale_, mvnShift_);
    //cerr<<"setImgDenseBox img cost:"<<timer1.toc()*1000<<" ms"<<endl;
    //timer1.tic();

    //std::cout << "densebox input" << 1 << " " << srcImg.channels() << " " << srcImg.rows << " " << srcImg.cols << std::endl;
    
    caffe_net->Forward(NULL);
    cudaDeviceSynchronize();
    //cerr<<"preprocess image cost:"<<timer1.toc()*1000<<" ms"<<endl;
    //timer1.tic();

    //const vector<Blob<float>*>& outBlob = caffe_net->ForwardPrefilled();
    caffe::Blob<float> *output_blob = caffe_net->blob_by_name("res_all").get();
    vector<int> outblob_shape = output_blob->shape();
    //cerr<<"caffe predict cost:"<<timer1.toc()*1000<<" ms"<<endl;
    //timer1.tic();

    int num = outblob_shape[0];//outBlob[0]->num();
    //CHECK_EQ(num, 1);

    const int map_height = outblob_shape[2];//outBlob[0]->height();
    const int map_width = outblob_shape[3];//outBlob[0]->width();
    const int map_size = map_height * map_width;
    const int scale_num = outblob_shape[1]/channel_per_scale_;//outBlob[0]->channels()/channel_per_scale_;

    const float pad_h_map = pad_h_/heat_map_a_;
    const float pad_w_map = pad_w_/heat_map_a_;

    //CHECK_EQ(outblob_shape[1]%channel_per_scale_,0);
    //CHECK_EQ((outblob_shape[1]/scale_num)/channel_per_scale_, class_num_);
    //CHECK_EQ((outblob_shape[1]/scale_num)%channel_per_scale_, 0);

    /**
     * Get bbox from botom[0]
     */
    vector< vector< RotateBBox<float> > > all_candidates;
    all_candidates.resize(class_num_);

    //const float* bottom_data = outputblobs[0]->cpu_data(); //outBlob[0]->mutable_cpu_data();
    const float* bottom_data = output_blob->cpu_data(); //outBlob[0]->mutable_cpu_data();
    for( int i = 0 ; i < num ; ++ i )
    {
        for( int scale_id = 0 ; scale_id < scale_num ; ++scale_id )
        {
            for( int class_id = 0 ; class_id < class_num_; ++class_id)
            {
                int score_channel = ((i*scale_num + scale_id)*class_num_ + class_id )* channel_per_scale_;;
                int offset_channel = score_channel + 1;

                const float* curScores = bottom_data + ((i * outblob_shape[1] + score_channel) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,score_channel  , 0,0);
                const float* dx1 =    bottom_data + ((i * outblob_shape[1] + offset_channel+0) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+0,0,0);
                const float* dy1 =    bottom_data + ((i * outblob_shape[1] + offset_channel+1) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+1,0,0);
                const float* dx2 =    bottom_data + ((i * outblob_shape[1] + offset_channel+2) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+2,0,0);
                const float* dy2 =    bottom_data + ((i * outblob_shape[1] + offset_channel+3) * outblob_shape[2] + 0) * outblob_shape[3] + 0;//outBlob[0]->offset(i,offset_channel+3,0,0);

                for(int off = 0; off< map_size; ++off)
                {
                    if(curScores[off] < nms_threshold_)
                        continue;
                    int h = off / map_width;
                    int w = off % map_width;
                    if(h >=  pad_h_map && h < map_height - pad_h_map && w >= pad_w_map  && w < map_width - pad_w_map)
                    {
                        float cur_h = h * heat_map_a_ + heat_map_b_ - pad_h_;
                        float cur_w = w * heat_map_a_ + heat_map_b_ - pad_w_;	

                        float left_top_x = cur_w - dx1[off]*heat_map_a_;
                        float left_top_y = cur_h - dy1[off]*heat_map_a_;
                        float right_bottom_x = cur_w - dx2[off]*heat_map_a_;
                        float right_bottom_y = cur_h - dy2[off]*heat_map_a_;

                        RotateBBox<float> candidate( left_top_x, left_top_y, right_bottom_x, right_bottom_y, curScores[off]);
                        all_candidates[class_id].push_back(candidate);
                    }
                }

            }
        }
    }

    //nms
    vector< vector< RotateBBox<float> > > output_candidates;
    output_candidates.resize(class_num_);
    for( int class_id = 0 ; class_id < class_num_ ; ++class_id )
    {
        vector< RotateBBox<float> >& cur_candidates = all_candidates[class_id];
        vector< RotateBBox<float> >& cur_output     = output_candidates[class_id];
        nms( cur_candidates, cur_output, (float)nms_overlap_ratio_, (int)nms_top_n_,false);
    }

    faces = output_candidates[0];

    //cerr<<"bbox process cost:"<<timer1.toc()*1000<<" ms"<<endl;
}

//detection rbox
bool PyramidDenseBox::predictPyramidDenseBox(caffe::shared_ptr<caffe::Net<float> > caffe_net, 
						Mat& img, 
						vector< RotateBBox<float> >& rotatedFaces)
{
    vector<Mat> pyramidImgs;
    //timer timer1;
    //timer1.tic();
    if(! constructPyramidImgs(img, pyramidImgs) )
    {
        //LOG(INFO)<< "predictDenseBoxPyramid:: constructPyramidImgs FAILED";
        return(false);
    }
    //cerr<<"build pyr cost:"<<timer1.toc()*1000<<" ms"<<endl;

    vector< RotateBBox<float> > detectedrotatedFaces;
	/*
	predictDenseBox(caffe_net, img, pyramidImgs, detectedrotatedFaces);
	*/

    for( int i = 0 ; i < pyramidImgs.size() ; ++i )
    {
        //timer1.tic();
        vector< RotateBBox<float> > faces;
        predictDenseBox( caffe_net, pyramidImgs[i], faces);
        //cerr<<"detect pyr["<<i<<"] cost:"<<timer1.toc()*1000<<" ms"<<endl;

        //back to the orignal image
        float ratio_x = float(img.cols) / pyramidImgs[i].cols;
        float ratio_y = float(img.rows) / pyramidImgs[i].rows;
        for ( int j = 0 ; j < faces.size() ; ++j )
        {
            faces[j].lt_x *= ratio_x;
            faces[j].lt_y *= ratio_y;
            faces[j].rt_x *= ratio_x;
            faces[j].rt_y *= ratio_y;
            faces[j].rb_x *= ratio_x;
            faces[j].rb_y *= ratio_y;                 			 
            faces[j].lb_x *= ratio_x;
            faces[j].lb_y *= ratio_y;                 			 
        }
        for( int j = 0 ; j < faces.size() ; ++j )
            detectedrotatedFaces.push_back( faces[j] );
    }

    rotatedFaces.clear();
    //timer1.tic();
    if( !nms( detectedrotatedFaces, rotatedFaces, (float)nms_overlap_ratio_, (int)nms_top_n_,false) )
        return(false);
    //cerr<<"nms cost:"<<timer1.toc()*1000<<" ms"<<endl;
    return(true);
}
bool PyramidDenseBox::predictPyramidDenseBox(caffe::shared_ptr<caffe::Net<float> > caffe_net, 
					const vector<Mat>& imgs, 
					vector<vector< RotateBBox<float> > >& rotatedFaces) {
	vector<vector<Mat> > pyramidImgs(imgs.size());
#pragma omp parallel for
	for(size_t i = 0; i < imgs.size(); ++i) {
		if(!constructPyramidImgs(imgs[i], pyramidImgs[i])) {
			cout << "construct Pyramid " << i << endl;
		}
		
	}	
	int pyramid_num = pyramidImgs[0].size();
	
	vector<vector<Mat> > batch_pyramidImgs(pyramid_num);
	for(size_t batch_i = 0; batch_i < pyramidImgs.size(); ++batch_i) {
		// CHECK_EQ(pyramid_num == pyramidImgs[batch_i].size());
		if(pyramid_num != pyramidImgs[batch_i].size()) {
			cout << "scale num not match "<< pyramidImgs[batch_i].size() << endl;
			return false;
		}
	}

	vector<vector<RotateBBox<float> > > all_scale_faces(pyramidImgs.size());
	for(size_t scale_i = 0; scale_i < pyramid_num; ++scale_i) {
		vector<Mat> one_scale_pyramid(imgs.size()); 
		for(size_t batch_i = 0; batch_i < pyramidImgs.size(); ++batch_i) {
			one_scale_pyramid[batch_i] = pyramidImgs[batch_i][scale_i];
		}
		
		vector<vector<RotateBBox<float> > > one_scale_faces;
		predictDenseBox(caffe_net, one_scale_pyramid, one_scale_faces);
		
		//back to the orignal image
		float ratio_x = float(imgs[0].cols) / one_scale_pyramid[0].cols;
		float ratio_y = float(imgs[0].rows) / one_scale_pyramid[0].rows;
		for(size_t batch_i = 0; batch_i < pyramidImgs.size(); ++batch_i) {
			vector<RotateBBox<float> >& one_img_bbox = one_scale_faces[batch_i];
			for(size_t box_i = 0; box_i < one_img_bbox.size(); ++box_i) {
				one_img_bbox[box_i].lt_x *= ratio_x;
				one_img_bbox[box_i].lt_y *= ratio_y;
				one_img_bbox[box_i].rt_x *= ratio_x;
				one_img_bbox[box_i].rt_y *= ratio_y;
				one_img_bbox[box_i].rb_x *= ratio_x;
				one_img_bbox[box_i].rb_y *= ratio_y;
				one_img_bbox[box_i].lb_x *= ratio_x;
				one_img_bbox[box_i].lb_y *= ratio_y;
			}
			all_scale_faces[batch_i].insert(all_scale_faces[batch_i].end(), 
							one_img_bbox.begin(), 
							one_img_bbox.end());
		}
	}
	rotatedFaces.resize(all_scale_faces.size());
	for(size_t batch_i = 0; batch_i < all_scale_faces.size(); ++batch_i) {
		if(!nms(all_scale_faces[batch_i],rotatedFaces[batch_i], (float)nms_overlap_ratio_, (int)nms_top_n_, false))
			return false;
	}
	return true;	
}

}
