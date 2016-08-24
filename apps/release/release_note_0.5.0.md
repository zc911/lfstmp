# Matrix Apps
### Version 0.4.4
2016-08-04

```
Update plate sdk to 2.2.4 and remove MAC check in watchdog
```

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.5.0 | 0.1.3 | 1.10 | * |

### Features
- New marker ssd model/window ssd model
- Send results to postresql
- Save images in a particular path
- Group pedestrian result
- Improve caffe lib
- Unify gpu plate color mapping with wentong plate color mapping

### Bugs
- DEEPVIDEO-522 打开Strorage且只用restful接口时，调用matrix奔溃
- DEEPVIDEO-498 matrix 识别symbols，整图与局部车辆图片识别不一致，不检测，也未正确识别遮阳板

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