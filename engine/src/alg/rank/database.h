//*****************************************************************************
// A great source code is like a poem that any comment shall profanes it.
//														-- devymex@gmail.com
//*****************************************************************************


#ifndef DATABASE_H_
#define DATABASE_H_

#include <vector>

namespace dg {


class CDatabase {
 public:
    enum QUERY_PARAMS {
        CUDA_BLOCKS = 50000,
        CUDA_THREADS = 500
    };

    struct DIST {
        int64_t id;            // Item index of one device
        float dist;        // Squared distance
    };

 protected:
    //
    int32_t m_nGPUs;
    int64_t m_nItemLen;        // Item length;
    int64_t m_nItemCnt;        // total number of database items
    int64_t m_nPosition;    // Data Loading Position (of one device)

    // Pointers of Device Memory
    std::vector<float *> m_ItemSets;
    std::vector<float *> m_QueryItem;
    std::vector<DIST *> m_QueryResults;

    std::vector<int> m_SampleIdx;

 public:
    // Constructor
    CDatabase();
    // Deconstructor
    virtual ~CDatabase();

    void SetWorkingGPUs(int32_t nGPUs);

    void Initialize(int64_t nItemCnt, int64_t nItemLen);

    void Clear();

    void AddItems(const float *pItems, const int64_t *pIds, int64_t nCnt);

    void ResetItems();

    void NearestN(const float *pItem, int64_t N, int64_t *pOutIds);

    int32_t GetGPUCount();

    int64_t GetItemCount();

    int64_t GetItemLength();

    void GetItem(int iGpu, int iPos, float *pItem, int64_t *pId);

 protected:
    void _DoQuery();

    void _GPUQuery(int64_t nCudaBlocks, int64_t nCudaThreads, int64_t nBaseIdx);

    void _UploadQueryItem(const float *pItem);

    void _DownloadResults(std::vector<DIST> &results, int64_t N);

    float _SampleMaxDist(int64_t N, int64_t nSamples);

    void _CountForGather(float fMaxDist, std::vector<int64_t> &gatherCnts);

    void _DumpResults(std::vector<DIST> &results);
};


void LoadDatabaseFromFile(const std::string &strFile, CDatabase &db);

}


#endif /* DATABASE_H_ */
