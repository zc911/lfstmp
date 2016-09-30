#ifndef _DGFACESDK_DATABASE_H_
#define _DGFACESDK_DATABASE_H_

#include <map>
#include <string>
#include "common.h"
#include <verification.h>
namespace DGFace{

typedef int FaceIdType;

class Database {
    public:
        virtual ~Database(void);
        void init(void);
        std::vector<FaceIdType> add(const std::vector<FeatureType> &feat_add);
        void del(const std::vector<FaceIdType>& del_ids);
        std::pair<float, int> search(const FeatureType &feat_query);
        void search(const FeatureType &feat_query, std::vector<std::pair<float, int> >& search_result);
    protected:
        Database(Verification* verifier);
        Verification*         _verifier;
        size_t                _feat_len;
        FaceIdType            _next_face_id;
        std::map<FaceIdType, FeatureType> _feat_db;
        virtual void load() = 0;
        virtual void save() = 0;
};

Database *create_database(const std::string &prefix = std::string());
}
#endif

