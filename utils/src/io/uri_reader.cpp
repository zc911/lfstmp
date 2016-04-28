#include <curl/curl.h>
#include <glog/logging.h>

#include "uri_reader.h"

using namespace std;

namespace dg
{

UriReader::reader_();

UriReader::UriReader()
{
    curl_global_init(CURL_GLOBAL_ALL);
}

UriReader::~UriReader()
{
    curl_global_cleanup();
}

size_t UriReader::write_callback(void* buffer, size_t size, size_t nmemb, void* stream)
{
    vector<uchar> *vector_buf = (vector<uchar> *)stream;
    uchar *tmp = (uchar *)buffer;
    int length = (size * nmemb) / sizeof(uchar);
    std::copy(tmp, tmp + length, std::back_inserter(vector_buf));
    return size * nmemb;
}

int UriReader::Read(const std::string uri, std::vector<uchar>& buffer)
{
    CURL *curl_handle = curl_easy_init();
    if (!curl_handle)
    {
        return -1;
    }

    curl_easy_setopt(curl_handle, CURLOPT_URL, uri.c_str());
    curl_easy_setopt(curl_handle, CURLOPT_VERBOSE, 1L); //0L for no verbose
    curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 1L); //oL for progress
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, UriReader::write_callback);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &buffer);
    CURLcode res = curl_easy_perform(curl_handle);
    if (CURLE_OK != res)
    {
        LOG(WARNING) << "curl perform failed: " << res << ", for uri " << uri;
        return -1;
    }

    url_easy_cleanup(curl_handle);
    return 0;
}

}