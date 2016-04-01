/*
 * restful_api.h
 *
 *  Created on: Feb 17, 2016
 *      Author: chenzhen
 */

#ifndef HTTP_CLIENT_H_
#define HTTP_CLIENT_H_
#include <curl/curl.h>

namespace deepglint {

class HttpClient {
 public:
    HttpClient() {
    }
    ;
    ~HttpClient() {
    }
    ;

    static size_t OnWriteData(void* buffer, size_t size, size_t nmemb,
                              void* lpVoid) {
        std::string* str = dynamic_cast<std::string*>((std::string *) lpVoid);
        if (NULL == str || NULL == buffer) {
            return -1;
        }

        char* pData = (char*) buffer;
        str->append(pData);
        return nmemb;
    }
    struct WriteThis {
        const char *readptr;
        int sizeleft;
    };

    static size_t read_callback(void *ptr, size_t size, size_t nmemb,
                                void *userp) {
        struct WriteThis *pooh = (struct WriteThis *) userp;

        if (size * nmemb < pooh->sizeleft) {
            *(char *) ptr = pooh->readptr[0]; /* copy one single byte */
            pooh->readptr++; /* advance pointer */
            pooh->sizeleft--; /* less data left */
            return 1; /* we return 1 byte at a time! */
        }

        return 0; /* no more data left to deliver */
    }

    int Post(const std::string & strUrl, const std::string & strPost,
             std::string & strResponse) {
        CURLcode res;
        curl_global_init (CURL_GLOBAL_ALL);
        CURL* curl = curl_easy_init();
        if (NULL == curl) {
            return CURLE_FAILED_INIT;
        }
        curl_easy_setopt(curl, CURLOPT_URL, strUrl.c_str());
        curl_slist *plist = curl_slist_append(NULL,
                                              "Content-Type:application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, plist);

        curl_easy_setopt(curl, CURLOPT_POST, 1);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strPost.c_str());
        //     // curl_easy_setopt(curl, CURLOPT_READFUNCTION, NULL);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, OnWriteData);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *) &strResponse);

        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3);
        res = curl_easy_perform(curl);
        switch (res) {
            case CURLE_OK:
                break;
            default:
                cout << "network error" << res << endl;
                LOG(WARNING) << "Network error " << res << endl;
                break;
        }
        curl_easy_cleanup(curl);
        return res;
    }
    ;
    int Get(const std::string & strUrl, std::string & strResponse) {
        CURLcode res;
        CURL* curl = curl_easy_init();
        if (NULL == curl) {
            return CURLE_FAILED_INIT;
        }
        curl_easy_setopt(curl, CURLOPT_URL, strUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, NULL);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, OnWriteData);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *) &strResponse);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        return res;
    }
    ;
};
}

#endif /* HTTP_CLIENT_H_ */
