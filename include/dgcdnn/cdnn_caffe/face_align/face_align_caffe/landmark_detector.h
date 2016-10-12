#pragma one
#include <vector>
#include <string>
class LandmarkDetector
{
    public:
        LandmarkDetector();
        LandmarkDetector(std::string& model_def1, std::string& weight_file1, std::string& model_def2, std::string& weight_file2 );
        ~LandmarkDetector();
        void init(std::string& model_def1, std::string& weight_file1, std::string& model_def2, std::string& weight_file2 );
        void detect(void* pImgData, int rows, int cols, int width_step,std::vector<float>& rbox, std::vector<float>& shape, std::vector<float>& confidence_scores);
    private:
        void* align_net;
        void* align_net_finelevel;
};

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
