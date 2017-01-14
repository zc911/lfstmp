# Matrix Apps
### Version 0.7.0
2017-01-13

```
同时支持人脸和车的相关功能
启用batch模式，大幅度提升人脸相关算法的吞吐量
支持人脸的ranker
支持CUDA7 和 CUDA8
重新整理人脸、车和车牌的模型及参数文件
```

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.7.0 | 0.1.5 | 1.10 | * |

### Feature
- 同时支持人脸和车的相关功能
- 启用batch模式，大幅度提升人脸相关算法的吞吐量
- 支持人脸的ranker
- 支持CUDA7 和 CUDA8
- 重新整理人脸、车和车牌的模型及参数文件


### Bugs



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
  "ProtocolType": "rpc|restful",
  "InstanceType": "witness",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6500,
    "Threads": [1]
  },
  "RankerType": "face",
  "Pack":{
    "Enable":false,
    "BatchSize":1,
    "TimeOut":100
  },
  "Feature": {
    "Vehicle": {
      "Enable": false,
      "EnableDetection": true,
      "EnableType": true,
      "EnableColor": true,
      "EnableGpuPlate": true,
      "EnablePlateEnhanced": false,
      "EnableMarker": true,
      "EnableDriverBelt": true,
      "EnableCoDriverBelt": true,
      "EnablePhone": true,
      "EnableFeatureVector": true,
      "EnableNonMotorVehicle": true,
      "EnablePedestrianAttr": true
    },
    "Face": {
      "Enable": true,
      "EnableDetection": true,
      "EnableAlignment": true,
      "EnableQuality": true,
      "EnableFeatureVector": true
    }
  },
  "Advanced": {
    "Detection": {
      "TargetMinSize": 400.0,
      "TargetMaxSize": 600.0
    },
    "PlateMxnet": {
      "BatchSize": 1,
      "EnableLocalProvince": false,
      "LocalProvinceConfidence": 0.6,
      "LocalProvinceText": "\u4eac"
    },
    "DriverPhone":{
      "Threshold": 0.9
    },
    "DriverBelt": {
      "Threshold": 0.9
    },
    "CoDriverBelt": {
      "Threshold": 0.9
    },
    "FaceDetect": {
      "Method": 3,
      "BatchSize": 8,
      "BodyRelativeFaceLeft": 0.2,
      "BodyRelativeFaceRight": 0.2,
      "BodyRelativeFaceTop": 0.2,
      "BodyRelativeFaceBottom": 6.0
    },
    "FaceAlignment":{
      "Method": 1,
      "Threshold": 0
    },
    "FaceQuality": {
      "BlurThreshold": 0
    },
    "FaceExtract": {
      "BatchSize": 1,
      "Method": 3,
      "BatchSize": 8,
      "EnableConcurrency": false
    },

    "Ranker": {
      "NormalizeAlpha": -0.02,
      "NormalizeBeta": 1.1,
      "Maximum": 1000000,
      "FeatureLen": 128,
      "SaveToFile": true,
      "SaveToFileIterval": 10000, // seconds
      "RepoPath": "./repo/shenyang",
      // must be absolutely path if use file://
      "ImageRootPath": "file:///home/chenzhen/Workspace/cpp/Matrix/apps/bin/Debug/repo/dgface/"
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
  "ModelPath":{
    "dgvehicle": "data/dgvehicle",
    "dgLP":"data/dgLP",
    "dgface":"data/dgface"
  }
}

```