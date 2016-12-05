#include <memory>
#include <database/db_simple.h>

using namespace std;
namespace DGFace{

SimpleDatabase::SimpleDatabase(const string& db_path, Verification* verifier)
    : Database(verifier), _db_path(db_path) {
}

SimpleDatabase::~SimpleDatabase(void) {
}

#pragma pack(push, 1)
struct FileFeatureEntry {
    FaceIdType      face_id;
    FeatureElemType feature_data[0];
};
#pragma pack(pop)

void SimpleDatabase::load() {
    FILE *fp = fopen(_db_path.c_str(), "rb");
    if(!fp){
        cerr << "File opening failed" << endl;
        return;
    }
    size_t count = fread(&_feat_len, sizeof(_feat_len), 1, fp);
    if (count < 1) {
        cerr << "File empty." << endl;
        _feat_len = 0;
        goto cleanup;
    }
    cout << "feature_len:" << _feat_len << endl;
    do {
        size_t feature_size = sizeof(FileFeatureEntry) + _feat_len * sizeof(FeatureElemType);
        unique_ptr<FileFeatureEntry> entry(reinterpret_cast<FileFeatureEntry *>(new char[feature_size]));
        while (true) {
            vector<FeatureElemType> feature;
            size_t count = fread(entry.get(), feature_size, 1, fp);
            if (count < 1)
                break;
            feature.resize(_feat_len);
            for (size_t i = 0; i < _feat_len; i++) {
                feature[i] = entry->feature_data[i];
            }
            _feat_db[entry->face_id] = feature;
            _next_face_id = max(_next_face_id, entry->face_id + 1);
        }
    } while (0);
cleanup:
    fclose(fp);
}

void SimpleDatabase::save() {
    FILE *fp = fopen(_db_path.c_str(), "wb");
    if(!fp){
        cerr << "File opening failed" << endl;
        return;
    }
    if (_feat_db.size() < 1) {
        cerr << "Feature db is empty." << endl;
        goto cleanup;
    }
    assert(_feat_len);
    cout << "feature_len:" << _feat_len << endl;
    fwrite(&_feat_len, sizeof(_feat_len), 1, fp);
    do {
        size_t feature_size = sizeof(FileFeatureEntry) + _feat_len * sizeof(FeatureElemType);
        unique_ptr<FileFeatureEntry> entry(reinterpret_cast<FileFeatureEntry *>(new char[feature_size]));
        for (auto iter : _feat_db) {
            entry->face_id = iter.first;
            for (size_t i = 0; i < _feat_len; i++) {
                entry->feature_data[i] = iter.second[i];
            }
            fwrite(entry.get(), feature_size, 1, fp);
        }
    } while (0);
cleanup:
    fclose(fp);
}

Database *create_database() {
    throw new runtime_error("database module will be removed");
}
/*
Database *create_database() {
	string prefix = "";
    Config *config = Config::instance();
    string type    = config->GetConfig<string>(prefix + "database", "simple");
    if (type == "simple") {
        string db_path = config->GetConfig<string>(prefix + "database.db_path", "feature_db.bin");
        Verification* verifier = create_verifier(verif_method::EUCLID);
        Database *db = new SimpleDatabase(db_path, verifier);
        db->init();
        return db;
    }
    throw new runtime_error("unknown database");
}
*/
}
