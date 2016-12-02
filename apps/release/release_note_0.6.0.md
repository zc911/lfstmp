# Matrix Apps
### Version 0.6.0
2016-12-02

```
同时支持CUDA 7.0 和CUDA 8.0
增加车辆相关的多个功能
更新多个模型，包括车辆检测、车身颜色、车牌、行人属性等模型
增加非机动车属性分类功能
移植车辆相关算法代码到SDK，修改Matrix调用算法的方式
使用conan管理Matrix的依赖库
```

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.6.0 | 0.1.5 | 1.10 | * |

### Feature
- DEEPVIDEO-691	升级Matrix车型车款model到1.2
- DEEPVIDEO-602	实现Single打包到Batch模式的缓冲机制
- DEEPVIDEO-601	更新行人属性model
- DEEPVIDEO-600	修改车辆的grpc接口描述，增加乘客、驾驶员、打电话、安全带等属性
- DEEPVIDEO-585	实现驾驶员打电话检测
- DEEPVIDEO-584	实现左右安全带的检测识别


### Bugs
- DEEPVIDEO-649	matrix grpc ranker无法完成getrankervector功能
- DEEPVIDEO-648 matrix grpc witnessservice 两个接口定义错误getindex,getindextxt
- DEEPVIDEO-610	Matrix 检测车辆，cutboard标识车辆有偏移，设别准确率也不同


### How to Install/Update
```
$ wget -O install.sh http://192.168.2.119/matrix/install.sh
$ chmod +x install.sh
$ ./install.sh 
```

### How to Run
```
$ sudo ./matrix_app [-port=$PORT] [-config=$CONFIG_FILE_PATH]
or
$ sudo ./matrix_app -help
for more help details 
```

### Config File
```json
{
  "Version": {
    "Code": "1.0.0",
    "Model": "1.9"
  },
  // "ProtocolType": "rpc",
  "ProtocolType": "rpc|restful",
  "InstanceType": "witness",
  // "InstanceType" : "ranker",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6500,
    "Threads": [1]
  },
  "Ranker": {
    "DefaultType": 2
  },
  "Feature": {
    "Vehicle": {
      "Enable": true,
      "EnableDetection": true,
      "EnableType": true,
      "EnableColor": true,
      "EnablePlate": false,
      "EnableGpuPlate": true,
      "EnablePlateEnhanced": false,
      "EnableMarker": true,
      "EnableFeatureVector": true,
      "EnablePedestrianAttr": true
    },
    "Face": {
      "Enable": false,
      "EnableDetection": false,
      "EnableFeatureVector": false
    }
  },
  "Advanced": {
    "Detection": {
      "Rescale": 400,
      "TargetMinSize": 300.0,
      "TargetMaxSize": 450.0,
      "CarOnly": false,
      "BatchSize": 1
    },
    "Color": {
      "BatchSize": 1,
      "ModelNum": 1
    },
    "Style": {
      "BatchSize": 1,
      "ModelNum": 1
    },
    "Plate": {
      "LocalProvince": "",
      "OCR": 1,
      "Locate": 5,
      "EnableSharp": false
    },
    "PlateMxnet": {
      "BatchSize": 1,
      "ImgStandardWidth": 400,
      "ImgStandardHeight": 600,
      "PlateStandardWidth": 300,
      "PlateStandardHeight": 100,
      "PlateNums": 2,
      "NumsProposal": 20,
      "EnableLocalProvince": true,
      "LocalProvinceText": "京",
      "LocalProvinceConfidence": 0.8
    },
    "Marker": {
      "MOTConfidence": 0.6,
      "BeltConfidence": 0.8,
      "GlobalConfidence": 0.8,
      "AccessoriesConfidence": 0.8,
      "OthersConfidence": 0.8,
      "TissueBoxConfidence": 0.8,
      "SunVisorConfidence": 0.8,
      "BatchSize": 1
    },
    "Window": {
      "BatchSize": 1
    },
    "FaceDetect": {
      "BatchSize": 1,
      "Confidence": 0.7,
      "BodyRelativeFaceLeft": 0.2,
      "BodyRelativeFaceRight": 0.2,
      "BodyRelativeFaceTop": 0.2,
      "BodyRelativeFaceBottom": 6.0,
      "Scale": 640
    },
    "FaceExtract": {
      "BatchSize": 1
    },
    "Ranker": {
      "Maximum": 100000
    },
    "ParseImageTimeout": 60
  },
  "Log": {
    "Dir": "",
    "Level": ""
  },
  "Render": {
    "Vehicle": {
      "Type": "data/mapping/vehicle_type.txt",
      "Model": "data/mapping/front_day_index_1_10.txt",
      "Color": "data/mapping/vehicle_color.txt",
      "Symbol": "data/mapping/vehicle_symbol.txt",
      "Plate": {
        "Color": "data/mapping/plate_color.txt",
        "ColorGpu": "data/mapping/plate_gpu_color.txt",
        "Type": "data/mapping/plate_type.txt"
      }
    },
    "Pedestrian": {
      "PedestrianAttr": "data/mapping/pedestrian_attr_type.txt",
      "PedestrianCategory": "data/mapping/pedestrian_attr_category.txt"
    }
  },
  "Debug": {
    "Enable": true,
    "EnableCutboard": true,
    "Encrypt": false
  },
  "Storage": {
    "Enabled": false,
    "Address": ["192.168.2.132:9877","192.168.2.119:9004"],
    "DBType": [0,1]
  },
  "DataPath": "data/data.dat"
}



```