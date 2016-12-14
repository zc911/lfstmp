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
- 增加检测阈值
- 增加逻辑：检测为车尾时不做小物件、安全带、打电话等特征的检测
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
  "ProtocolType": "restful",
  //   "ProtocolType": "rpc|restful",
  "InstanceType": "witness",
  // "InstanceType" : "ranker",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6500,
    "Threads": [
      1
    ]
  },
  "RankerType": "car|face",
  "Pack": {
    "Enable": false,
    "BatchSize": 1,
    "TimeOut": 100
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
      "EnableDriverBelt": false,
      "EnableCoDriverBelt": true,
      "EnablePhone": true,
      "EnableFeatureVector": true,
      "EnableNonMotorVehicle": true,
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
      "TargetMinSize": 400.0,
      "TargetMaxSize": 600.0,
      "CarOnly": false,
      "BatchSize": 1,
      "Threshold": 0.5
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
      "EnableLocalProvince": false,
      "ImgStandardHeight": 600,
      "ImgStandardWidth": 400,
      "LocalProvinceConfidence": 0.6,
      "LocalProvinceText": "\u4eac",
      "NumsProposal": 20,
      "PlateNums": 4,
      "PlateStandardHeight": 150,
      "PlateStandardWidth": 250
    },
    "Marker": {
      "MOTConfidence": 0.6,
      "BeltConfidence": 0.8,
      "GlobalConfidence": 0.8,
      "AccessoriesConfidence": 0.8,
      "OthersConfidence": 0.8,
      "TissueBoxConfidence": 0.8,
      "SunVisorConfidence": 0.8,
      "BatchSize": 2
    },
    "DriverBelt": {
      "BatchSize": 2,
      "Threshold": 0.0
    },
    "CoDriverBelt": {
      "BatchSize": 2,
      "Threshold": 0.0
    },
    "DriverPhone": {
      "BatchSize": 2
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
    "NonMotorVehicle": {
      "NonMotorAttr": "data/mapping/bitri_attr_type.txt",
      "NonMotorCategory": "data/mapping/bitri_attr_category.txt"
    },
    "Pedestrian": {
      "PedestrianAttr": "data/mapping/pedestrian_attr_type.txt",
      "PedestrianCategory": "data/mapping/pedestrian_attr_category.txt"
    },
    "Vehicle": {
      "Color": "data/mapping/vehicle_color.txt",
      "Model": "data/mapping/vehicle_style_v1.12_4063.txt",
      "Plate": {
        "Color": "data/mapping/plate_color.txt",
        "ColorGpu": "data/mapping/plate_gpu_color.txt",
        "Type": "data/mapping/plate_type.txt"
      },
      "Symbol": "data/mapping/vehicle_symbol.txt",
      "Type": "data/mapping/vehicle_type.txt"
    }
  },
  "Debug": {
    "Enable": true,
    "EnableCutboard": true,
    "Encrypt": false,
    "Visualization": false
  },
  "Storage": {
    "Address": [
      "192.168.2.119:9004",
      "192.168.2.132:9877",
      "./"
    ],
    "DBType": [
      0,
      1,
      2
    ],
    "Enabled": false
  },
  //"DataPath": "data/data_tollgate.dat"
  "DataPath": "data/data.dat"
}


```