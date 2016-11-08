#include <database.h>

using namespace std;
namespace DGFace{

Database::Database (Verification* verifier) : _verifier(verifier), _feat_len(0), _next_face_id(0) {
}

Database::~Database (void) {
    delete _verifier; 
}

void Database::init() {
    load();
}

vector<FaceIdType> Database::add(const vector<FeatureType> &feat_add) {
    pair <float, int> tmp;
    bool aflag = false;
    int num_add = 0;
    vector<FaceIdType> result;
    result.reserve(feat_add.size());
    if (!feat_add.size()) {
        cerr << "No new items added" << endl;
        return result;
    }
    for(auto &feat_query : feat_add) {
        if (_feat_len == 0) {
            _feat_len = feat_query.size();
        } else {
            assert(_feat_len == feat_query.size());
        }
        tmp = search(feat_query); 
        // if cannot find the exactly same image
        if (tmp.first < 1 - 1e-5 ) {
            _feat_db[_next_face_id] = feat_query;
            num_add += 1;
            cout << "add new id to db" << endl;
            result.push_back(_next_face_id);
            _next_face_id++;
            aflag = true;
        } else {
            cout << "id already exist in db" << endl;
            result.push_back(tmp.second);
        }
    }
    // update the feat_db.bin
    if (aflag) {
        save();
    }
    cout << num_add << " added" << endl;
    return result;
}

void Database::del(const vector<FaceIdType>& del_ids) {
    bool dflag = false;
    int num_del = 0;
    if (!del_ids.size()) {
        cerr << "No items deleted";                                                          
        return;
    }
    for(auto del_id: del_ids) {
        auto iter = _feat_db.find(del_id);
        if (iter != _feat_db.end()) {
            _feat_db.erase(iter);
            num_del += 1;
            dflag = true;
        }
    }
    // update the feat_db.bin
    if (dflag) {
        save();
    }
    cout << num_del << " deleted" << endl;
}

pair<float, int> Database::search(const FeatureType &feat_query) {
    int id = -1;
    float max_simi = -1.0f;

    for(auto entry : _feat_db) {
        float simi = _verifier->verify(feat_query, entry.second);
        if (simi > max_simi) {
            max_simi = simi;
            id = entry.first;
        }
    }
    return make_pair(max_simi, id);
}

void Database::search(const FeatureType &feat_query, vector< pair<float, int> >& search_result)
{
    search_result.resize(0);
    for(auto entry: _feat_db)
    {
        float simi = _verifier->verify(feat_query, entry.second);
        search_result.push_back(make_pair(-simi, entry.first));
    }
    sort(search_result.begin(), search_result.end());
    for (auto &item : search_result) {
        item.first = -item.first;
    }
}
}

