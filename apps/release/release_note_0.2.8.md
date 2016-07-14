# Matrix Apps
### Version 0.2.8
2016-07-13

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.2.0 | 0.1.3 | 1.10 | * |

### Features
- Modify from multiple threads to multiple processes because of performance requirement
- Improve witness performance
- Use deepv detection instead of SSD detection because of multi-thread bug in SSD
- Improve plate performance by some parameter changed
- Change the release folder structure. Puts all the dependency libraries to folder libs
and put all the model file(encrypted) into folder libs/1
- Add batch_run.sh script to run multiple matrix_apps processes
- Modify config file
    ```
    "Threads": [1,6] //means 1 thread on GPU 0 and 6 threads on GPU 1
    "CarOnly": true, //use car only detection instead of multi-detection
    ```
- Modify config default value
    ```
        "Color": {
          "BatchSize": 8,
          "ModelNum": 1 // from 2 to 1
        }
        "PlateMxnet": {
          "BatchSize": 8,
          "ImgStandardWidth": 400,
          "ImgStandardHeight": 600,
          "PlateStandardWidth": 300,
          "PlateStandardHeight": 100,
          "PlateNums": 2,
          "NumsProposal": 20
        },
    ```

### Bugs fixed
- Plate config reversed

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
  //  "ProtocolType": "rpc",
  "ProtocolType": "restful",
  "InstanceType": "witness",
  //"InstanceType" : "ranker",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6500,
    "Threads": [1,6] //means 1 thread on GPU 0 and 6 threads on GPU 1
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
      "EnableMarker": false,
      "EnableFeatureVector": false,
      "EnablePedestrianAttr": false
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
      "CarOnly": true, //use car only detection instead of multi-detection
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
      "NumsProposal": 20
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
      "Color": "data/mapping/vehicle_colorhaoquan.txt",
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