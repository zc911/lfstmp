// ------------------------------------------------------------------
// Written by Zhang Wang(wangzhang@outlook.com)
// ------------------------------------------------------------------

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <math.h>

#include "caffe/fast_rcnn_layers.hpp"
#include <sys/time.h>
#include <opencv2/opencv.hpp>
using namespace cv;

namespace caffe {
template<typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    ProposalParameter proposal_param = this->layer_param_.proposal_param();
    feat_stride = proposal_param.feat_stride();

}

template<typename Dtype>
void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
}

template<typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {

    cudaDeviceSynchronize();
    struct timeval time_start;
    gettimeofday(&time_start, NULL);

    struct timeval time_hh;
    gettimeofday(&time_hh, NULL);
    float hh_timecost = (time_hh.tv_sec - time_start.tv_sec)
            + double(time_hh.tv_usec - time_start.tv_usec) / 1.0e6;
    const Dtype* scores = bottom[0]->cpu_data();
    const Dtype* bbox_deltas = bottom[1]->cpu_data();
    const Dtype* im_info = bottom[2]->cpu_data();
    const Dtype* img_data = bottom[3]->cpu_data();

    const vector<int> & shape0 = bottom[0]->shape();
    const vector<int> & shape1 = bottom[1]->shape();

    const int count_0 = bottom[0]->count();
    const int count_1 = bottom[1]->count();

    int min_size = 16;

    std::vector<std::vector<int> > anchor = { { -20.75, -7., 35.75, 22. }, {
            -32.65863992, -13.21320344, 47.65863992, 28.21320344 }, { -49.5,
            -22., 64.5, 37. }, { -73.31727984, -34.42640687, 88.31727984,
            49.42640687 }, { -107., -52., 122., 67. }, { -154.63455967,
            -76.85281374, 169.63455967, 91.85281374 }, { -222., -112., 237.,
            127. }, { -12., -12., 27., 27. }, { -20.28427125, -20.28427125,
            35.28427125, 35.28427125 }, { -32., -32., 47., 47. }, {
            -48.56854249, -48.56854249, 63.56854249, 63.56854249 }, { -72, -72,
            87, 87 },
            { -105.13708499, -105.13708499, 120.13708499, 120.13708499 }, {
                    -152, -152, 167, 167 }, { -5.75, -19.5, 20.75, 34.5 }, {
                    -11.44543648, -30.89087297, 26.4454364, 45.89087297 }, {
                    -19.5, -47., 34.5, 62. }, { -30.89087297, -69.78174593,
                    45.89087297, 84.78174593 }, { -47., -102., 62., 117. }, {
                    -69.78174593, -147.56349186, 84.78174593, 162.56349186 }, {
                    -102., -212., 117., 227. } };
    // arange
    std::vector<int> shift_x;
    std::vector<int> shift_y;
    int start = 0;
    int step = 1;
    for (int value = start; value < width_; value += step)
        shift_x.push_back(value * feat_stride);
    for (int value = start; value < height_; value += step)
        shift_y.push_back(value * feat_stride);

    // meshgrid

    int x_l = shift_x.size();
    int y_l = shift_y.size();

    std::vector<std::vector<int> > shift(y_l * x_l, vector<int>(4));
    int count_y = 0;
    for (int i = 0; i < x_l * y_l; i++) {
        for (int j = 0; j < 4; j++) {
            if ((j == 0) || (j == 2)) {
                shift[i][j] = shift_x[i % x_l];
            }

            if ((j == 1) || (j == 3)) {

                if ((i % x_l == 0) && (j == 1)) {
                    count_y += 1;
                }
                shift[i][j] = shift_y[count_y - 1];
            }
        }
    }

    /*
     * generate shifted anchor
     */
    int A = anchor.size();
    int K = shift.size();

    std::vector<std::vector<int> > anchor_t(K * A, vector<int>(4));
    std::vector<std::vector<int> > shift_t(K * A, vector<int>(4));
    std::vector<std::vector<int> > shifted_anchor(K * A, vector<int>(4));

    for (int i = 0; i < K * A; i++) {
        anchor_t[i] = anchor[i % A];
    }
    int count_s = -1;
    for (int i = 0; i < K * A; i++) {
        if (i % A == 0) {
            count_s += 1;
        }
        shift_t[i] = shift[count_s];

    }
    for (int i = 0; i < K * A; i++) {
        for (int j = 0; j < 4; j++) {
            shifted_anchor[i][j] = anchor_t[i][j] + shift_t[i][j];
        }
    }

    /*
     * reshape bbox_deltas as (1 * H * W * A x 4)  and clip predicted boxes to image
     */
    std::vector<Dtype> scores0;
    std::vector<Dtype> scores_;
    int H_W = shape0[2] * shape0[3];
    int count_all = 0;
    int period = -1;

    for (int i = 0; i < count_0 / 2; i++) {
        if (count_all % shape0[1] == 0) {
            period += 1;
            count_all = shape0[1] / 2;
        };
        scores0.push_back(scores[period + count_all * H_W]);

        count_all += 1;
    }

    struct timeval time_pre;
    gettimeofday(&time_pre, NULL);

    float pre_timecost = (time_pre.tv_sec - time_start.tv_sec)
            + double(time_pre.tv_usec - time_start.tv_usec) / 1.0e6;

    std::vector<std::vector<Dtype> > bbox_deltas_(count_1 / 4,
                                                  vector<Dtype>(4));
    std::vector<std::vector<Dtype> > proposals(count_1 / 4, vector<Dtype>(4));
    std::vector<std::vector<Dtype> > proposals_;

    int H_W2 = shape1[2] * shape1[3];
    int count_all2 = 0;
    int period2 = -1;
    int widths, heights, ctr_x, ctr_y;
    Dtype dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h, ws, hs;
    for (int i = 0; i < count_1 / 4; i++) {
        if (scores0[i] < 0.9)
            continue;
        for (int j = 0; j < 4; j++) {
            if (count_all2 % shape1[1] == 0) {
                period2 += 1;
            }
            bbox_deltas_[i][j] = bbox_deltas[period2
                    + (count_all2 % shape1[1]) * H_W2];
            count_all2 += 1;
        }

        // bbox transform inv
        widths = shifted_anchor[i][2] - shifted_anchor[i][0] + 1.0;
        heights = shifted_anchor[i][3] - shifted_anchor[i][1] + 1.0;
        ctr_x = shifted_anchor[i][0] + 0.5 * widths;
        ctr_y = shifted_anchor[i][1] + 0.5 * heights;
        dx = bbox_deltas_[i][0];
        dy = bbox_deltas_[i][1];
        dw = bbox_deltas_[i][2];
        dh = bbox_deltas_[i][3];
        pred_ctr_x = dx * widths + ctr_x;
        pred_ctr_y = dy * heights + ctr_y;
        pred_w = exp(dw) * widths;
        pred_h = exp(dh) * heights;

        proposals[i][0] = std::max(
                std::min(Dtype(pred_ctr_x - 0.5 * pred_w),
                         Dtype(im_info[1] - 1)),
                Dtype(0));
        proposals[i][1] = std::max(
                std::min(Dtype(pred_ctr_y - 0.5 * pred_h),
                         Dtype(im_info[0] - 1)),
                Dtype(0));
        proposals[i][2] = std::max(
                std::min(Dtype(pred_ctr_x + 0.5 * pred_w),
                         Dtype(im_info[1] - 1)),
                Dtype(0));
        proposals[i][3] = std::max(
                std::min(Dtype(pred_ctr_y + 0.5 * pred_h),
                         Dtype(im_info[0] - 1)),
                Dtype(0));
        // remove predicted boxes with either height or width < threshod
        ws = proposals[i][2] - proposals[i][0] + 1.0;
        hs = proposals[i][3] - proposals[i][1] + 1.0;
        if ((ws >= min_size * im_info[2]) && (hs >= min_size * im_info[2])) {
            proposals_.push_back(proposals[i]);
            scores_.push_back(scores0[i]);
        }
    }

    struct timeval time_mid;
    gettimeofday(&time_mid, NULL);

    float mid_timecost = (time_mid.tv_sec - time_start.tv_sec)
            + double(time_mid.tv_usec - time_start.tv_usec) / 1.0e6;

    /*
     * sort all (proposal, score) by score from highest to lowest
     * take top pre_nms_topN(6,000)
     */

    vector<size_t> idx(scores_.size());
    for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;

    sort(idx.begin(), idx.end(),
         [scores_](size_t i1, size_t i2) {return scores_[i1] > scores_[i2];});

    std::vector<std::vector<Dtype> > proposals_f;
    std::vector<Dtype> scores_f;
    // only calculate 1000 proposals
    for (size_t i = 0; i < std::min(Dtype(50), Dtype(idx.size())); i++) {
        //if(scores_[idx[i]] > 0.9) {
        if (1) {
            proposals_f.push_back(proposals_[idx[i]]);
            scores_f.push_back(scores_[idx[i]]);
        } else {
            break;
        }
    }

    /*
     * apply nms
     * take after_nms_topN
     * return the top proposals
     */
    // nms
    Dtype nms_thresh = 0.7;
    std::vector<std::vector<Dtype> > proposals_nms;
    std::vector<std::vector<Dtype> > proposals_tmp = proposals_f;
    std::vector<std::vector<Dtype> > proposals_tmp2 = proposals_f;
    std::vector<Dtype> scores_nms;
    Dtype xx1, xx2, yy1, yy2, w, h, inter, ovr;
    int count_temp = 0;
    while (proposals_tmp.size() > 0 && count_temp < 100) {
        proposals_tmp2.clear();
        proposals_nms.push_back(proposals_tmp[0]);
        for (int i = 1; i < proposals_tmp.size(); i++) {
            xx1 = std::max(proposals_tmp[0][0], proposals_tmp[i][0]);
            yy1 = std::max(proposals_tmp[0][1], proposals_tmp[i][1]);
            xx2 = std::min(proposals_tmp[0][2], proposals_tmp[i][2]);
            yy2 = std::min(proposals_tmp[0][3], proposals_tmp[i][3]);
            w = std::max(Dtype(0.0), xx2 - xx1 + 1);
            h = std::max(Dtype(0.0), yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter
                    / ((proposals_tmp[0][2] - proposals_tmp[0][0] + 1)
                            * (proposals_tmp[0][3] - proposals_tmp[0][1] + 1)
                            + (proposals_tmp[i][2] - proposals_tmp[i][0] + 1)
                                    * (proposals_tmp[i][3] - proposals_tmp[i][1]
                                            + 1) - inter);
            if (ovr <= nms_thresh) {
                proposals_tmp2.push_back(proposals_tmp[i]);
            }
        }
        proposals_tmp.clear();
        proposals_tmp = proposals_tmp2;
        count_temp++;
    }

    if (proposals_nms.size() == 0) {
        vector<Dtype> tmp;
        tmp.push_back(1);
        tmp.push_back(2);
        tmp.push_back(3);
        tmp.push_back(4);
        proposals_nms.push_back(tmp);
    }

    vector<int> top_shape;  // = {proposals_nms.size(), 5};
    top_shape.push_back(proposals_nms.size());
    top_shape.push_back(5);
    top[0]->Reshape(top_shape);
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int i = 0; i < proposals_nms.size(); i++) {
        top_data[i * 5 + 0] = Dtype(0);
        top_data[i * 5 + 1] = proposals_nms[i][0];
        top_data[i * 5 + 2] = proposals_nms[i][1];
        top_data[i * 5 + 3] = proposals_nms[i][2];
        top_data[i * 5 + 4] = proposals_nms[i][3];
    }

    struct timeval time_end;
    gettimeofday(&time_end, NULL);

    float timecost = (time_end.tv_sec - time_start.tv_sec)
            + double(time_end.tv_usec - time_start.tv_usec) / 1.0e6;
}

template<typename Dtype>
void ProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);
}
