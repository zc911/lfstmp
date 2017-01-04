from conans import ConanFile, CMake

class DgFace(ConanFile):
    name = "dgface"
    version = "1.0.0"
    settings = "os", "compiler", "build_type", "arch"
    exports = "*"
    url = "https://github.com/deepglint/dgface.git"
    license = "deepglint"

    def package(self):
        self.copy("*", dst="include", src="include")
        self.copy("*", dst="lib", src="lib")
        self.copy('release_note_%s.md'%self.version)

    def package_info(self):
        self.cpp_info.libs = ["dgface"]


