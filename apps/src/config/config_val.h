/*
 * config_val.h
 *
 *  Created on: Apr 25, 2016
 *      Author: chenzhen
 */

#ifndef CONFIG_VAL_H_
#define CONFIG_VAL_H_

#include <string>

using namespace std;

namespace dg {

class ConfigValue {
 public:
    const static string VEHICLE_MODEL_MAPPING_FILE;
    const static string VEHICLE_COLOR_MAPPING_FILE;
    const static string VEHICLE_SYMBOL_MAPPING_FILE;
    const static string VEHICLE_PLATE_COLOR_MAPPING_FILE;
    const static string VEHICLE_PLATE_TYPE_MAPPING_FILE;
};

const string ConfigValue::VEHICLE_MODEL_MAPPING_FILE = "Render/Vehicle/Model";
const string ConfigValue::VEHICLE_COLOR_MAPPING_FILE = "Render/Vehicle/Color";
const string ConfigValue::VEHICLE_SYMBOL_MAPPING_FILE = "Render/Vehicle/Symbol";
const string ConfigValue::VEHICLE_PLATE_COLOR_MAPPING_FILE = "Render/Vehicle/Plate/Color";
const string ConfigValue::VEHICLE_PLATE_TYPE_MAPPING_FILE = "Render/Vehicle/Plate/Type";

}

#endif /* CONFIG_VAL_H_ */
