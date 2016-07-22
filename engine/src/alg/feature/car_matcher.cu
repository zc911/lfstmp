#include <glog/logging.h>
#include "car_matcher.h"

namespace dg {
#define FEATURE_NUM_CUDA 256
#define MAX_IMG_NUM 100000

#define CUDA_CALL(value) {  \
cudaError_t _m_cudaStat = value;    \
if (_m_cudaStat != cudaSuccess) {   \
    fprintf(stderr, "Error %s at line %d in file %s\n", \
            cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);    \
}}

struct box {
    ushort x;
    ushort y;
    ushort height;
    ushort width;
};

CarMatcher::CarMatcher() {
    feature_num_ = FEATURE_NUM_CUDA;
    max_resize_size_ = 300;
    max_mis_match_ = 50;
    min_remarkableness_ = 0.8;
    max_mapping_offset_ = 50;
    selected_area_weight_ = 50;
    min_score_thr_ = 100;
    profile_time_ = false;

    cudaStreamCreate(&stream_);
    CUDA_CALL(
        cudaMallocManaged(&query_pos_cuda_, FEATURE_NUM_CUDA * sizeof(ushort) * 2, cudaMemAttachHost));
    CUDA_CALL(
        cudaMallocManaged(&query_desc_cuda_, FEATURE_NUM_CUDA * sizeof(uint) * 8, cudaMemAttachHost));
    CUDA_CALL(
        cudaMallocManaged(&db_pos_cuda_, FEATURE_NUM_CUDA * MAX_IMG_NUM * sizeof(ushort) * 2, cudaMemAttachHost));
    CUDA_CALL(
        cudaMallocManaged(&db_desc_cuda_, FEATURE_NUM_CUDA * MAX_IMG_NUM * sizeof(uint) * 8, cudaMemAttachHost));
    CUDA_CALL(
        cudaMallocManaged(&db_width_cuda_, MAX_IMG_NUM * sizeof(ushort), cudaMemAttachHost));
    CUDA_CALL(
        cudaMallocManaged(&db_height_cuda_, MAX_IMG_NUM * sizeof(ushort), cudaMemAttachHost));
    CUDA_CALL(
        cudaMallocManaged(&score_cuda_, MAX_IMG_NUM * sizeof(int), cudaMemAttachHost));
}

CarMatcher::~CarMatcher() {
	CUDA_CALL(cudaFree(query_pos_cuda_));
    CUDA_CALL(cudaFree(query_desc_cuda_));
    CUDA_CALL(cudaFree(db_pos_cuda_));
    CUDA_CALL(cudaFree(db_desc_cuda_));
    CUDA_CALL(cudaFree(db_width_cuda_));
    CUDA_CALL(cudaFree(db_height_cuda_));
    CUDA_CALL(cudaFree(score_cuda_));
    CUDA_CALL(cudaStreamDestroy(stream_));
}

__global__ void compute_match_score_kernel(box query_box, ushort *query_pos, uint *query_desc, 
					ushort *db_pos, uint *db_desc, ushort query_width, ushort query_height, 
					ushort *db_width, ushort *db_height, int max_resize_size, int feature_num, 
					float min_remarkableness, int max_mis_match, int selected_area_weight, int max_mapping_offset, int *score) {
    //Calculate new bounding box
    box query_box_resize;
    box db_box_resize;
    if(query_box.x > 0) {
	    float resize_rto_query = 0;
	    float resize_rto_db = 0;
        if(query_width > query_height)
		    resize_rto_query = (float)max_resize_size / (float)query_width;
		else resize_rto_query = (float)max_resize_size / (float)query_height;
	    query_box_resize.x = (float)query_box.x * resize_rto_query;
	    query_box_resize.y = (float)query_box.y * resize_rto_query;
	    query_box_resize.width = (float)query_box.width * resize_rto_query;
	    query_box_resize.height = (float)query_box.height * resize_rto_query;
        if(db_width[blockIdx.x] > db_height[blockIdx.x])
		    resize_rto_db = (float)max_resize_size / (float)db_width[blockIdx.x];
		else resize_rto_db = (float)max_resize_size / (float)db_height[blockIdx.x];
	    db_box_resize.x = (float)query_box.x * resize_rto_db - max_mapping_offset;
	    db_box_resize.y = (float)query_box.y * resize_rto_db - max_mapping_offset;
	    db_box_resize.width = (float)query_box.width * resize_rto_db + max_mapping_offset * 2;
	    db_box_resize.height = (float)query_box.height * resize_rto_db + max_mapping_offset * 2;
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
    //if(blockIdx.x == 638 && threadIdx.x == 0) {
    //    printf("GPU %d %d %d %d\n", query_box_resize.x, query_box_resize.y, query_box_resize.width, query_box_resize.height);
    //    printf("GPU %d %d %d %d\n", db_box_resize.x, db_box_resize.y, db_box_resize.width, db_box_resize.height);
    //}
    float max_mapping_offset_rto = (float) max_mapping_offset/(float) max_resize_size;
	max_mapping_offset_rto = 2 * max_mapping_offset_rto * max_mapping_offset_rto;
    //Calculate the score
	int min_dist = INT_MAX, sec_dist = INT_MAX;
	int min_idx = -1;
    int score_tmp;
    for (int i = 0; i < FEATURE_NUM_CUDA; i++) {
        if(db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + i * 2] == -1)
		    break;
		float pos1_x_rto = (float)query_pos[threadIdx.x * 2] / (float)query_width;
		float pos1_y_rto = (float)query_pos[threadIdx.x * 2 + 1] / (float)query_height;
		float pos2_x_rto = (float)db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + i * 2] / (float)db_width[blockIdx.x];
		float pos2_y_rto = (float)db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + i * 2 + 1] / (float)db_height[blockIdx.x];
	    if ((pos2_x_rto - pos1_x_rto) * (pos2_x_rto - pos1_x_rto) + (pos2_y_rto - pos1_y_rto) * (pos2_y_rto - pos1_y_rto) < max_mapping_offset_rto) {
	        score_tmp = 0;
	    	for (int j = 0; j < 8; j++) {
          	    score_tmp += __popc(query_desc[threadIdx.x * 8 + j] ^ db_desc[blockIdx.x * feature_num * 8 + i * 8 + j]);
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
	}
    score_tmp = 0;
	if ((min_dist <= (unsigned int) (min_remarkableness * sec_dist))
			&& (min_dist <= (unsigned int) max_mis_match)) {
		if (query_pos[threadIdx.x * 2] >= query_box_resize.x &&
				query_pos[threadIdx.x * 2] <= (query_box_resize.x + query_box_resize.width) &&
				query_pos[threadIdx.x * 2 + 1] >= query_box_resize.y &&
				query_pos[threadIdx.x * 2 + 1] <= (query_box_resize.y + query_box_resize.height) &&
			    db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2] >= db_box_resize.x &&
			    db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2] <= db_box_resize.x + db_box_resize.width &&
			    db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2 + 1] >= db_box_resize.y &&
			    db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2 + 1] <= db_box_resize.y + db_box_resize.height) {
			//if(blockIdx.x == 638)
			//	printf("*GPU above thr %d min_idx %d query pos %d %d res pos %d %d\n", threadIdx.x, min_idx, query_pos[threadIdx.x * 2], query_pos[threadIdx.x * 2+1], db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2], db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2+1]);
		    score_tmp = score_tmp + selected_area_weight;
		}
		else {
			//if(blockIdx.x == 638)
			//	printf("GPU under thr %d min_idx %d query pos %d %d res pos %d %d\n", threadIdx.x, min_idx, query_pos[threadIdx.x * 2], query_pos[threadIdx.x * 2+1], db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2], db_pos[blockIdx.x * FEATURE_NUM_CUDA * 2 + min_idx * 2+1]);
			score_tmp++;
		}
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


vector<int> CarMatcher::computeMatchScoreGpu(
    const CarRankFeature &des, const Rect &in_box,
    const vector<CarRankFeature> &all_des) {

    box query_box;
    query_box.x = in_box.x;
    query_box.y = in_box.y;
    query_box.width = in_box.width;
    query_box.height = in_box.height;
    for (int j = 0; j < feature_num_; j++) {
        if (j < des.position_.rows) {
            query_pos_cuda_[j * 2 + 0] = des.position_.at<ushort>(j, 0);
            query_pos_cuda_[j * 2 + 1] = des.position_.at<ushort>(j, 1);
            for (int k = 0; k < 32; k++)
                query_desc_cuda_[j * 32 + k] = des.descriptor_.at<uchar>(j, k);
        } else {
            query_pos_cuda_[j * 2 + 0] = -1;
            query_pos_cuda_[j * 2 + 1] = -1;
            for (int k = 0; k < 32; k++)
                query_desc_cuda_[j * 32 + k] = 0;
        }
    }
    ushort query_width = des.width_;
    ushort query_height = des.height_;
    for (int i = 0; i < all_des.size(); i++) {
        db_width_cuda_[i] = all_des[i].width_;
        db_height_cuda_[i] = all_des[i].height_;
        for (int j = 0; j < feature_num_; j++) {
            if (j < all_des[i].position_.rows) {
                db_pos_cuda_[i * feature_num_ * 2 + j * 2 + 0] = all_des[i]
                    .position_.at<ushort>(j, 0);
                db_pos_cuda_[i * feature_num_ * 2 + j * 2 + 1] = all_des[i]
                    .position_.at<ushort>(j, 1);
                for (int k = 0; k < 32; k++)
                    db_desc_cuda_[i * feature_num_ * 32 + j * 32 + k] =
                        all_des[i].descriptor_.at<uchar>(j, k);
            } else {
                db_pos_cuda_[i * feature_num_ * 2 + j * 2 + 0] = -1;
                db_pos_cuda_[i * feature_num_ * 2 + j * 2 + 1] = -1;
                for (int k = 0; k < 32; k++)
                    db_desc_cuda_[i * feature_num_ * 32 + j * 32 + k] = 0;
            }
        }
    }

    CUDA_CALL(cudaStreamAttachMemAsync(stream_, query_pos_cuda_));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, query_desc_cuda_));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, db_pos_cuda_));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, db_desc_cuda_));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, db_width_cuda_));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, db_height_cuda_));
    CUDA_CALL(cudaStreamAttachMemAsync(stream_, score_cuda_));

    dim3 grid, block;
    grid = dim3(all_des.size());
    block = dim3(FEATURE_NUM_CUDA);
	compute_match_score_kernel << < grid, block, 0, stream_ >> > (query_box, query_pos_cuda_, (uint *) query_desc_cuda_,
        db_pos_cuda_, (uint *) db_desc_cuda_, query_width,
        query_height, db_width_cuda_, db_height_cuda_,
        max_resize_size_, feature_num_, min_remarkableness_,
        max_mis_match_, selected_area_weight_, max_mapping_offset_, score_cuda_);
    CUDA_CALL(cudaStreamSynchronize(stream_));
    
    CUDA_CALL(cudaGetLastError());

    return vector<int>(score_cuda_, score_cuda_ + all_des.size());
}
}

