#include <curl/curl.h>
#include <glog/logging.h>
#include <iterator>
#include <iostream>

#include "uri_reader.h"
#include "string_util.h"
using namespace std;

namespace dg {

//UriReader::reader_();

UriReader::UriReader() {
    curl_global_init(CURL_GLOBAL_ALL);
    cout << "initialize success" << endl;
}

UriReader::~UriReader() {
    curl_global_cleanup();
}

size_t write_callback(void *buffer, size_t size, size_t nmemb, void *stream) {
    vector<uchar> *vec_buffer = (vector<uchar> *) stream;
    uchar *tmp = (uchar *) buffer;
    int length = (size * nmemb) / sizeof(uchar);
    std::copy(tmp, tmp + length, std::back_inserter(*vec_buffer));
    return size * nmemb;
}

int UriReader::Read(const std::string uri, std::vector<uchar> &buffer) {
    CURL *curl_handle = curl_easy_init();
    if (!curl_handle) {
        return -1;
    }

    // trim the spaces before or after the string.
    string theUri = uri;
    trimLR(theUri);

    curl_easy_setopt(curl_handle, CURLOPT_URL, theUri.c_str());
    curl_easy_setopt(curl_handle, CURLOPT_VERBOSE, 0L); //0L for no verbose
    curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 1L); //oL for progress
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &buffer);
    CURLcode res = curl_easy_perform(curl_handle);

    curl_easy_cleanup(curl_handle);

    if (CURLE_OK != res) {
        LOG(ERROR) << "curl perform failed: " << res << ", for uri " << uri;
        return -1;
    }

    if (buffer.size() == 0) {
        LOG(ERROR) << "load image size is zero" << uri;
        return -1;
    }


    return 0;
}

}
