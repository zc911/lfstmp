# Matrix Apps
### Version 0.4.3
2016-08-01

```
Update plate sdk to 2.2.3 
```

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.4.3 | 0.1.3 | 1.10 | * |

### Features

- Update plate sdk to 2.2.3, use one thread instead of multiple threads in case bug


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
  "ProtocolType": "restful",
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
      "TargetMinSize": 400.0,
      "TargetMaxSize": 600.0,
      "CarOnly": false, 
      "BatchSize": 8 
    },
    "Color": {
      "BatchSize": 8,
      "ModelNum": 1 
    },
    "Style": {
      "BatchSize": 8,
      "ModelNum": 1
    },
    "Plate": {
      "LocalProvince": "",
      "OCR": 1,
      "Locate": 5,
      "EnableSharp": false
    },
    "PlateMxnet": {
      "BatchSize": 8,
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
      "BatchSize": 8
    },
    "Window": {
      "BatchSize": 8
    },
    "FaceDetect": {
      "BatchSize": 1,
      "Confidence": 0.7,
      "Scale": 640
    },
    "FaceExtract": {
      "BatchSize": 1
    },
    "Ranker": {
      "Maximum": 100000
    }
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
      },
      "PedestrianAttr": "data/mapping/pedestrian_attr_type.txt"
    }
  },
  "Debug": {
    "Enable": true,
    "EnableCutboard": true,
    "Encrypt": false
  },
  "Storage": {
    "Enabled": false,
    "Address": "192.168.2.119:9004",
    "DBType": 2
  },
  "DataPath": "data/data.dat"
}


```