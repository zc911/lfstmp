#include "../dog.h"
#include <iostream>
#include <string>
#include <glog/logging.h>

extern "C" {
#include "hasp_api.h"
}

using namespace std;

unsigned char vendor_code[] =
        "h0oQsRm1H3k5QCqadqGcWa32kYCwykmVhadKxornbi7gTjNJQR6e2FDiIjzwFQ7rft/bxX0mvuElya"
                "ulK7kGSCVFCpPkcZRCP1I38FPlAk4ljHXoYU37pIvK6p4naSTChJHXUrBPLiE11LCvNYf5G2IVOTWX"
                "rE4+tiy96boPYCericjml0TzyZpbBY7jtFKSdiazaRI+K1m4lcK9ztNsIAFe4Hi4Iu78HMEWuUIiph"
                "hsgLT+rC0xh3GMUo3I1t4Jbz69kXuC7rR0NJQZCc4fcz0EHXv9ukydjbNHcF1y+xusQ8FmjyC5HW4j"
                "d9aKhUa9qHmLsPrT+BrpVXmZjOhUSOFeZqZIVXfSYW6VecVZVIaVnV0GqW8aGDs/PSAkkLpPmr9y7n"
                "QT6p+p+sRbE2rRv07EYw7cU0MFNRIjX62xjmfkwknDa9tuPWaKaM9lQ4UqC1IvObvLIQQBm1vJV8Va"
                "jtPkddgsS3uLYOtpPwhdxITPXDb753TcWFGCQDCFWmdx9xCEgsOvZsRzOBdQvLQaBFFTlQiE20Uglu"
                "Xu9668rEJwUiS9LG4y5EYaodvqmjEX1oS2EApPFw6h1+JoS/Mwi6tzdkXn/vsxOlArgeBWM0XwJKzt"
                "YgpekVkZzfSAImjPxtOOFkX9jCodhpdpqMSkazOeqtF5oYOIVo8gOmZHQUmZyTbzlzYUXkuDyFmuzZ"
                "KSTGsKpdLiCDrLMPv3+ogPcxWXBs7/aLtQmEYUcPVvsbBnvI9klgFtoEmHgH5sRJt5DtoeSDvW1Xf4"
                "1C+ezPn1cSrw+RSu2XsyhMbYlcVPTpmEg4IRtIpn3k+V//DX0EnZaV1E/MsgR8ifYOiuVa1l80Y3X4"
                "FyK9I3De3NXPMum4TfIfXPhW13gWEDFeuY1KXqO0K6Cx97i6o7OA2nonD2FDJrnmuWfvGkv4oWwUD2"
                "k1t5Krpn/g2LNPuEoaTpVVzMJBWJNQKgrRxg2DDeipaIMw==";

hasp_handle_t handle;
hasp_status_t status;

KeyStatus loginDog(const int keyId) {

    KeyStatus ret = KEY_OK;
    status = hasp_login(1, (hasp_vendor_code_t *) vendor_code, &handle);
    switch (status) {

        case HASP_STATUS_OK:
            break;

        case HASP_FEATURE_NOT_FOUND: {
            ret = KEY_ERROR;
            break;
        }
        case HASP_CONTAINER_NOT_FOUND: {
            ret = KEY_ERROR;
            break;
        }

        case HASP_OLD_DRIVER: {
            ret = KEY_SYS_ERROR;
            break;
        }
        case HASP_NO_DRIVER: {
            ret = KEY_SYS_ERROR;
            break;
        }
        case HASP_INV_VCODE: {
            ret = KEY_PW_INVALID;
            break;
        }
        default: {
            ret = KEY_ERROR;
            break;
        }
    }
    if (ret != KEY_OK) {
        LOG(ERROR)<< "Login dog failed: " << status << endl;
    }

    return ret;
}

KeyStatus logoutDog(const int keyId) {
    KeyStatus ret = KEY_OK;
    status = hasp_logout(handle);

    switch (status) {
        case HASP_STATUS_OK: {
            break;
        }
        case HASP_INV_HND: {
            ret = KEY_ERROR;
            break;
        }
        default: {
            ret = KEY_ERROR;
        }
    }

//    if (ret != KEY_OK) {
//        LOG(ERROR)<< "Logout dog failed: " << status << endl;
//    }

    return ret;
}

KeyStatus writeDog(const int keyId, const int offset, const unsigned char *src,
                   int len) {

    KeyStatus ret = KEY_OK;
    status = hasp_write(handle, HASP_FILEID_RW, offset, len, src);

    if (status != HASP_STATUS_OK) {
        ret = KEY_ERROR;
        LOG(ERROR)<< "Write dog failed: " << status << endl;
        logoutDog(keyId);
    }

    return ret;
}

KeyStatus readDog(const int keyId, const int offset, unsigned char *dest,
                  int len) {

    KeyStatus ret = KEY_OK;
    status = hasp_read(handle, HASP_FILEID_RW, offset, len, dest);

    if (status != HASP_STATUS_OK) {
        LOG(ERROR)<< "Read dog failed: " << status << endl;
        ret = KEY_ERROR;
        logoutDog(keyId);
    }

    return ret;
}

KeyStatus encryptByDog(const int dogId, unsigned char *data, int len) {
    KeyStatus ret = KEY_OK;
    status = hasp_encrypt(handle, data, len);
    if (status != HASP_STATUS_OK) {
        DLOG(ERROR)<< "Encrypt by dog failed: " << status << endl;
        ret = KEY_ERROR;
        logoutDog(dogId);
    }

    return ret;
}
KeyStatus decryptByDog(const int dogId, unsigned char *data, int len) {
    KeyStatus ret = KEY_OK;
    status = hasp_decrypt(handle, data, len);
    if (status != HASP_STATUS_OK) {
        DLOG(ERROR)<< "Decrypt by dog failed: " << status << endl;
        ret = KEY_ERROR;
        logoutDog(dogId);
    }

    return ret;
}
