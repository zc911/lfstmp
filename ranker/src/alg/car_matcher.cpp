/*
 * car_matcher.cpp
 *
 *  Created on: Feb 17, 2016
 *      Author: haoquan
 */

#include <glog/logging.h>
#include "car_matcher.h"

#define FEATURE_NUM_CUDA 256
#define MAX_IMG_NUM 10000

CarMatcher::CarMatcher() {
    feature_num_ = FEATURE_NUM_CUDA;
    orb_ = ORB(feature_num_);
    max_resize_size_ = 300;
    max_mis_match_ = 50;
    min_remarkableness_ = 0.8;
    max_mapping_offset_ = 50;
    selected_area_weight_ = 50;
    profile_time_ = false;

#if USE_CUDA
    cudaStreamCreate(&stream_);
    CUDA_CALL(cudaMallocManaged(&query_pos_cuda, FEATURE_NUM_CUDA * sizeof(ushort) * 2, cudaMemAttachHost));
    CUDA_CALL(cudaMallocManaged(&query_desc_cuda, FEATURE_NUM_CUDA * sizeof(uint) * 8, cudaMemAttachHost));
    CUDA_CALL(cudaMallocManaged(&db_pos_cuda, FEATURE_NUM_CUDA * MAX_IMG_NUM * sizeof(ushort) * 2, cudaMemAttachHost));
    CUDA_CALL(cudaMallocManaged(&db_desc_cuda, FEATURE_NUM_CUDA * MAX_IMG_NUM * sizeof(uint) * 8, cudaMemAttachHost));
    CUDA_CALL(cudaMallocManaged(&db_width_cuda, MAX_IMG_NUM * sizeof(ushort), cudaMemAttachHost));
    CUDA_CALL(cudaMallocManaged(&db_height_cuda, MAX_IMG_NUM * sizeof(ushort), cudaMemAttachHost));
    CUDA_CALL(cudaMallocManaged(&score_cuda, MAX_IMG_NUM * sizeof(int), cudaMemAttachHost));
#endif
}

CarMatcher::~CarMatcher()
{
#if USE_CUDA
    CUDA_CALL(cudaStreamDestroy(stream_));
#endif
}


void CarMatcher::extract_descriptor(const Mat &img, CarDescriptor &des) {
    if (profile_time_)
        t_profiler_feature_.Reset();
    des.car_height = img.rows;
    des.car_width = img.cols;
    Mat resize_img;
    Size new_size;
    calc_new_size(des.car_height, des.car_width, new_size);
    if (img.channels() != 3)
        LOG(WARNING)<<"Color image is required.";
    if ((img.rows < 10) || (img.cols < 10))
        LOG(WARNING)<<"Image needs to be larger than 10*10 to extract enough feature.";
    resize(img, resize_img, new_size);
    orb_(resize_img, Mat(), key_point_, descriptor_);
    if (key_point_.size() < 50)
        LOG(WARNING)<<"Not enough feature extracted.";
    descriptor_.copyTo(des.descriptor);
    des.position = Mat::zeros(key_point_.size(), 2, CV_16UC1);
    for (int i = 0; i < key_point_.size(); i++) {
        des.position.at<ushort>(i, 0) = ((ushort) key_point_[i].pt.x);
        des.position.at<ushort>(i, 1) = ((ushort) key_point_[i].pt.y);
    }
    if (profile_time_) {
        t_profiler_str_ = "Descriptor";
        t_profiler_feature_.Update(t_profiler_str_);
    }
}

int CarMatcher::compute_match_score(const CarDescriptor &des1, const CarDescriptor &des2, const Rect &box) {
    if (profile_time_)
        t_profiler_matching_.Reset();
    Rect box1, box2;
    calc_new_box(des1, des2, box, box1, box2);
    int score = 0;
    for (int i = 0; i < des1.descriptor.rows; i++) {
        uint min_dist = 9999;
        uint sec_dist = 9999;
        int min_idx = -1, sec_idx = -1;
        const uchar* query_feat = des1.descriptor.ptr<uchar>(i);
        for (int j = 0; j < des2.descriptor.rows; j++)
            if (calc_dis2(des1.position.at<ushort>(i, 0),
                    des1.position.at<ushort>(i, 1),
                    des2.position.at<ushort>(j, 0),
                    des2.position.at<ushort>(j, 1))
                    < max_mapping_offset_ * max_mapping_offset_) {
                const uchar* train_feat = des2.descriptor.ptr(j);
                uint dist = calc_hamming_distance(query_feat, train_feat);
                if (dist < min_dist) {
                    sec_dist = min_dist;
                    sec_idx = min_idx;
                    min_dist = dist;
                    min_idx = j;
                } else if (dist < sec_dist) {
                    sec_dist = dist;
                    sec_idx = j;
                }
            }
        if ((min_dist <= (unsigned int) (min_remarkableness_ * sec_dist))
                && (min_dist <= (unsigned int) max_mis_match_)) {
            if ((is_in_box(des1.position.at<ushort>(i, 0),
                    des1.position.at<ushort>(i, 1), box1))
                    && (is_in_box(des2.position.at<ushort>(min_idx, 0),
                            des2.position.at<ushort>(min_idx, 1), box2))) {
                score = score + selected_area_weight_;
            } else
                score++;
        }
    }
    if (profile_time_) {
        t_profiler_str_ = "Matching";
        t_profiler_matching_.Update(t_profiler_str_);
    }
    return score;
}

void CarMatcher::calc_new_box(
    const CarDescriptor &des1,
    const CarDescriptor &des2, 
    const Rect &box, Rect &box1, Rect &box2) {
    if (box.x > 0) {
        float resize_rto1 = max(des1.car_height, des1.car_width);
        resize_rto1 = ((float) max_resize_size_) / resize_rto1;
        box1.x = box.x * resize_rto1;
        box1.y = box.y * resize_rto1;
        box1.width = box.width * resize_rto1;
        box1.height = box.height * resize_rto1;
        float resize_rto2 = max(des2.car_height, des2.car_width);
        resize_rto2 = ((float) max_resize_size_) / resize_rto2;
        box2.x = box.x * resize_rto2 - max_mapping_offset_;
        box2.y = box.y * resize_rto2 - max_mapping_offset_;
        box2.width = box.width * resize_rto2 + max_mapping_offset_ * 2;
        box2.height = box.height * resize_rto2 + max_mapping_offset_ * 2;
    } else {
        box1.x = 0;
        box1.y = 0;
        box1.width = max_resize_size_;
        box1.height = max_resize_size_;
        box2.x = 0;
        box2.y = 0;
        box2.width = max_resize_size_;
        box2.height = max_resize_size_;
    }
}

vector<int> CarMatcher::compute_match_score_cpu(const CarDescriptor &des, const Rect &in_box, const vector<CarDescriptor> &all_des)
{
    vector<int> score;
    for(int i = 0; i < all_des.size(); i ++)
    {
        score.push_back(compute_match_score(des, all_des[i], in_box));
    }
    return score;
}

vector<int> CarMatcher::compute_match_score(const CarDescriptor &des, const Rect &in_box, const vector<CarDescriptor> &all_des)
{
#if USE_CUDA
    return compute_match_score_gpu(des, in_box, all_des);
#else
    return compute_match_score_cpu(des, in_box, all_des);
#endif
}

#if USE_CUDA
__global__ void compute_match_score_kernel(box query_box, ushort *query_pos, uint *query_desc, 
    ushort *db_pos, uint *db_desc, 
    ushort query_width, ushort query_height, 
    ushort *db_width, ushort *db_height, 
    int max_resize_size, int feature_num, 
    float min_remarkableness, int max_mis_match, 
    int selected_area_weight, int *score) {
    //Calculate new bounding box
    box query_box_resize;
    box db_box_resize;
    if(query_box.x > 0) {
        float resize_rto_query = 0;
        float resize_rto_db = 0;
        if(query_width > query_height)
            resize_rto_query = (float)max_resize_size / query_width;
        else resize_rto_query = (float)max_resize_size / query_height;
        query_box_resize.x = query_box.x * resize_rto_query;
        query_box_resize.y = query_box.y * resize_rto_query;
        query_box_resize.width = query_box.width * resize_rto_query;
        query_box_resize.height = query_box.height * resize_rto_query;
        if(db_width[blockIdx.x] > db_height[blockIdx.x])
            resize_rto_db = (float)max_resize_size / db_width[blockIdx.x];
        else resize_rto_db = (float)max_resize_size /db_height[blockIdx.x];
        db_box_resize.x = query_box.x * resize_rto_db;
        db_box_resize.y = query_box.y * resize_rto_db;
        db_box_resize.width = query_box.width * resize_rto_db;
        db_box_resize.height = query_box.height * resize_rto_db;
    } else {
        query_box_resize.x = 0;
        query_box_resize.y = 0;
        query_box_resize.width = max_resize_size;
        query_box_resize.height = max_resize_size;
        db_box_resize.x = 0;
        db_box_resize.y = 0;
        db_box_resize.width = max_resize_size;
        db_box_resize.height = max_resize_size;
    }
    //Calculate the score
    int min_dist = INT_MAX, sec_dist = INT_MAX;
    int min_idx = -1;
    int score_tmp;
    for (int i = 0; i < FEATURE_NUM_CUDA; i++) {
        //__syncthreads();
        if(db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + i * 2] == -1)
            break;
        score_tmp = 0;
        for (int j = 0; j < 8; j++) {
                score_tmp += __popc(query_desc[threadIdx.x * 8 + j] ^ db_desc[blockIdx.x * feature_num* 8 + i * 8 + j]);
        }
        if (score_tmp < min_dist) {
            sec_dist = min_dist;
            min_dist = score_tmp;
            min_idx = i;
        }
        else if (score_tmp < sec_dist) {
            sec_dist = score_tmp;
        }
    }
    score_tmp = 0;
    if ((min_dist < (unsigned int) (min_remarkableness * sec_dist))
            && (min_dist < (unsigned int) max_mis_match)) {
        if (query_pos[threadIdx.x * 2] > query_box_resize.x &&
                query_pos[threadIdx.x * 2] < (query_box_resize.x + query_box_resize.width) &&
                query_pos[threadIdx.x * 2 + 1] > query_box_resize.y &&
                query_pos[threadIdx.x * 2 + 1] < (query_box_resize.y + query_box_resize.height) &&
                db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2] > db_box_resize.x &&
                db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2] < db_box_resize.x + db_box_resize.width &&
                db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2 + 1] > db_box_resize.y &&
                db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2 + 1] < db_box_resize.y + db_box_resize.height) {
            score_tmp = score_tmp + selected_area_weight;
        }
        else score_tmp++;
    }
    if(query_pos[threadIdx.x * 2] == -1)
        score_tmp = 0;
    __shared__ int score_shared[FEATURE_NUM_CUDA];
    score_shared[threadIdx.x] = score_tmp;
    // Sum the score of each image in the shared memory
    for (int stride = blockDim.x; stride > 1; stride >>=1) {
        __syncthreads();
        if (stride % 2 != 0 && threadIdx.x == 0)
            score_shared[0] += score_shared[stride - 1];
        if (threadIdx.x < (stride / 2))
            score_shared[threadIdx.x] += score_shared[threadIdx.x + stride / 2];
    }
    __syncthreads();
    // Write to the global memory
    if (threadIdx.x == 0)
        score[blockIdx.x] = score_shared[0];
}

vector<int> CarMatcher::compute_match_score_gpu(
    const CarDescriptor &des, 
    const Rect &in_box, 
    const vector<CarDescriptor> &all_des) {


    box query_box;
    query_box.x = in_box.x;
    query_box.y = in_box.y;
    query_box.width = in_box.width;
    query_box.height = in_box.height;
    for (int j = 0; j < feature_num_; j++) {
        if (j < des.position.rows) {
            query_pos_cuda[j * 2 + 0] = des.position.at<ushort>(j, 0);
            query_pos_cuda[j * 2 + 1] = des.position.at<ushort>(j, 1);
            for (int k = 0; k < 32; k++)
                query_desc_cuda[j * 32 + k] = des.descriptor.at<uchar>(j, k);
        } else {
            query_pos_cuda[j * 2 + 0] = -1;
            query_pos_cuda[j * 2 + 1] = -1;
            for (int k = 0; k < 32; k++)
                query_desc_cuda[j * 32 + k] = 0;
        }
    }
    ushort query_width = des.car_width;
    ushort query_height = des.car_height;
    for (int i = 0; i < all_des.size(); i++) {
        db_width_cuda[i] = all_des[i].car_width;
        db_height_cuda[i] = all_des[i].car_height;
        for (int j = 0; j < feature_num_; j++) {
            if (j < all_des[i].position.rows) {
                db_pos_cuda[i * feature_num_ * 2 + j * 2 + 0] = all_des[i].position.at<ushort>(j, 0);
                db_pos_cuda[i * feature_num_ * 2 + j * 2 + 1] = all_des[i].position.at<ushort>(j, 1);
                for (int k = 0; k < 32; k++)
                    db_desc_cuda[i * feature_num_ * 32 + j * 32 + k] = all_des[i].descriptor.at<uchar>(j, k);
            } else {
                db_pos_cuda[i * feature_num_ * 2 + j * 2 + 0] = -1;
                db_pos_cuda[i * feature_num_ * 2 + j * 2 + 1] = -1;
                for (int k = 0; k < 32; k++)
                    db_desc_cuda[i * feature_num_ * 32 + j * 32 + k] = 0;
            }
        }
    }
    
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, query_pos_cuda));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, query_desc_cuda));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, db_pos_cuda));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, db_desc_cuda));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, db_width_cuda));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, db_height_cuda));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, score_cuda));

    dim3 grid, block;
    grid = dim3(all_des.size());
    block = dim3(FEATURE_NUM_CUDA);
    compute_match_score_kernel<<<grid, block, 0, stream_>>>(query_box, query_pos_cuda, (uint*)query_desc_cuda, 
        db_pos_cuda, (uint*)db_desc_cuda, query_width, 
        query_height, db_width_cuda, db_height_cuda, 
        max_resize_size_, feature_num_, min_remarkableness_, 
        max_mis_match_, selected_area_weight_, score_cuda);
    CUDA_CALL(cudaStreamSynchronize(stream_));
    CUDA_CALL(cudaGetLastError());

    return vector<int>(score_cuda, score_cuda + all_des.size());
}
#endif
