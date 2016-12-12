#include "config.h"

#ifndef SRC_CHECK_FILE_EXIST_H_
#define SRC_CHECK_FILE_EXIST_H_

using namespace std;
using namespace dg;

namespace dg {

bool FilesAllExist(Config *config) {
    auto fileExist = [](const char *pathName)->bool {
        if (access(pathName, F_OK) == -1) {
            DLOG(ERROR) << "Can't find file : " << pathName << endl;
            return false;
        }
        return true;
    };
    bool flag = true;
    flag = flag && fileExist(((string)config->Value(VEHICLE_MODEL_MAPPING_FILE)).c_str());
    flag = flag && fileExist(((string)config->Value(VEHICLE_COLOR_MAPPING_FILE)).c_str());
    flag = flag && fileExist(((string)config->Value(VEHICLE_SYMBOL_MAPPING_FILE)).c_str());
    flag = flag && fileExist(((string)config->Value(VEHICLE_PLATE_COLOR_MAPPING_FILE)).c_str());
    flag = flag && fileExist(((string)config->Value(VEHICLE_PLATE_TYPE_MAPPING_FILE)).c_str());
    flag = flag && fileExist(((string)config->Value(VEHICLE_TYPE_MAPPING_FILE)).c_str());
    flag = flag && fileExist(((string)config->Value(PEDESTRIAN_ATTR_TYPE)).c_str());
    flag = flag && fileExist(((string)config->Value(PEDESTRIAN_ATTR_CATAGORY)).c_str());
    flag = flag && fileExist(((string)config->Value(RENDER_VEHICLE_TYPE)).c_str());
    flag = flag && fileExist(((string)config->Value(RENDER_VEHICLE_MODEL)).c_str());
    flag = flag && fileExist(((string)config->Value(RENDER_VEHICLE_COLOR)).c_str());
    flag = flag && fileExist(((string)config->Value(RENDER_VEHICLE_SYMBOL)).c_str());
    flag = flag && fileExist(((string)config->Value(RENDER_VEHICLE_PLATE_COLOR)).c_str());
    flag = flag && fileExist(((string)config->Value(RENDER_VEHICLE_PLATE_GPU_COLOR)).c_str());
    flag = flag && fileExist(((string)config->Value(RENDER_VEHICLE_PLATE_TYPE)).c_str());

    flag = flag && fileExist(((string)config->Value(DATAPATH)).c_str());

    bool is_encrypted = (bool) config->Value(DEBUG_MODEL_ENCRYPT);
    string modelPath = "data/" + (is_encrypted ? string("1/") : string("0/"));

    string data_config_path = (string) config->Value(DATAPATH);
    string json_data = ReadStringFromFile(data_config_path, "r");
    Config *dataConfig = new Config();
    dataConfig->LoadString(json_data);


 /*   flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_COLOR_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_COLOR_DEPLOY_MODEL)).c_str());

    int model_num = (int) config->Value(ADVANCED_STYLE_MODEL_NUM);
    model_num = model_num == 0 ? 1 : model_num;
    for (int i = 0; i < model_num; ++i) {
        flag = flag && fileExist(((modelPath)
            + (string) dataConfig->Value(FILE_STYLE_TRAINED_MODEL) + std::to_string(i) + ".dat").c_str());
        flag = flag && fileExist(((modelPath)
            + (string) dataConfig->Value(FILE_STYLE_DEPLOY_MODEL) + std::to_string(i) + ".txt").c_str());
    }

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_DETECTION_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_DETECTION_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_CAR_ONLY_DETECTION_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_CAR_ONLY_DETECTION_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_CAR_ONLY_CONFIRM_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_CAR_ONLY_CONFIRM_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_ACCELERATE_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_ACCELERATE_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_MARKER_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_MARKER_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_MARKER_ONLY_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_MARKER_ONLY_DEPLOY_MODEL)).c_str()); 
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_DRIVER_BELT_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_DRIVER_BELT_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_CODRIVER_BELT_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_CODRIVER_BELT_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_DRIVER_PHONE_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_DRIVER_PHONE_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PEDESTRIAN_ATTR_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PEDESTRIAN_ATTR_DEPLOY_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PEDESTRIAN_ATTR_TAGNAME_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_WINDOW_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_WINDOW_DEPLOY_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_WINDOW_ONLY_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_WINDOW_ONLY_DEPLOY_MODEL)).c_str());


    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_FACE_DETECT_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_FACE_DETECT_DEPLOY_MODEL)).c_str());

    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_FACE_EXTRACT_TRAINED_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_FACE_EXTRACT_DEPLOY_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_FACE_EXTRACT_ALIGN_MODEL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_FACE_EXTRACT_ALIGN_DEPLOY)).c_str()); */


    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_FCN_SYMBOL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_FCN_PARAM)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_RPN_SYMBOL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_RPN_PARAM)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_POLYREG_SYMBOL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_POLYREG_PARAM)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_ROIP_SYMBOL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_ROIP_PARAM)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_CHRECOG_SYMBOL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_CHRECOG_PARAM)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_COLOR_SYMBOL)).c_str());
    flag = flag && fileExist((modelPath + (string)dataConfig->Value(FILE_PLATE_COLOR_PARAM)).c_str());

    delete dataConfig;

    return flag;
}

}

#endif /* SRC_CHECK_FILE_EXIST_H_ */
