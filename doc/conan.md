## conan 使用方法

1.安装conan
```
pip install conan
```
2.查看帮助
```
conan -h
```
3.配置conan
```
conan remote remove conan
conan remote add conan_server http://192.168.2.21:9300
```
4.将编好的库传到本地conan
首先新建conanfiile.py,假设要编mxnet库,conanfile.py中写入以下内容
```
from conans import ConanFile, CMake

class Mxnet(ConanFile): 
    name = "Mxnet"           # 名字
    version = "0.1.0"        # 版本号
    settings = "os", "compiler", "build_type", "arch" 
    exports = "*"
    
    def package(self):
        self.copy("*", dst="include", src="include")
        self.copy("*", dst="lib", src="lib")

    def package_info(self):
        self.cpp_info.libs = ["mxnet"]
```
在同一目录下include目录和lib目录,将所需的头文件和库文件放入即可
目录结构如下所示
```
include/
├── base.h
├── c_api.h
├── c_predict_api.h
├── engine.h
├── io.h
├── kvstore.h
├── mxrtc.h
├── ndarray.h
├── operator.h
├── operator_util.h
├── optimizer.h
├── resource.h
├── storage.h
└── symbolic.h

lib/
└── Linux-x86_64
    └── libmxnet.so

```
最后运行如下命令即可将编好的库放入conan中
```
conan export deepglint/stable
```


5.查看本地conan库
```
conan search
```
6.将本地的库上传到远程服务器
```
conan upload Hello/0.1@deepglint/stable --all -r=conan_server
```
7.查看远程conan库
```
conan search -r=conan_server
```
8.使用conan
在CMakeLists.txt同一目录下新建conanfile.txt,requires下为依赖的库
命名格式为`库名字/版本号@用户/版本`
```
[requires]
Cudnn/4.0.7@deepglint/stable
Dgcaffe/0.1.0@deepglint/stable_cuda_7.0
Dgmedia/1.8.0@deepglint/stable

[generators]
cmake
```
然后进入build目录执行命令
```
conan install .. # 第一次执行`conan install .. --build`
```
然后执行
```
cmake ..
```
即可

