from conans import ConanFile, CMake

class DGFace(ConanFile):
	name = "DGFace"
	version = "1.0.0"
	setting = "os", "compiler", "build_type", "arch"
	exports = "*"

def package(self):
	self.copy("*", dst="include", src="include")
	self.copy("*", dst="lib", src="lib")

def package_info(self):
	self.cpp_info.libs = [facesdk]
