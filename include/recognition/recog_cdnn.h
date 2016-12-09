#ifndef _dgfacesdk_recognition_cdnn_h_
#define _dgfacesdk_recognition_cdnn_h_
#include <string>
#include <recognition.h>
#include "ExtractMPMetricFeature.h"


namespace DGFace{
class CdnnRecog: public Recognition {
	public:
		CdnnRecog(std::string configPath, std::string modelDir, bool multi_thread, bool is_encrypt);
		CdnnRecog(const std::string& model_dir, bool multi_thread, bool is_encrypt);
		virtual ~CdnnRecog();

		void recog_impl(const std::vector<cv::Mat>& faces, 
			const std::vector<AlignResult>& alignment,
			std::vector<RecogResult>& results);

	private:
		Cdnn::MPMetricFeature::ExtractMPMetricFeature _extractor;
        bool _multi_thread;
};
}
#endif
