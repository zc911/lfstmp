#define __cplusplus 201103L

#include <algorithm>
#include <fstream>
#include <future>
#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

//#include "mytimer.h"
#include "database.h"

namespace dg {
//***************************************************************************//
//**  Error Handling Macros and Function									 //
//***************************************************************************//
//-----------------------------------------------------------------------------
#define CUABORT(msg)                        \
{                                            \
    cuPrintError(msg, __FILE__, __LINE__);    \
    exit(-1);                                \
}
//-----------------------------------------------------------------------------
#define CUASSERT(exp)                        \
{                                            \
    if (!(exp))                                \
    {                                        \
        CUABORT(#exp);                        \
    }                                        \
}
//-----------------------------------------------------------------------------
#define CUCHECK(exp)                        \
{                                            \
    cudaError_t err = (exp);                \
    if (err != cudaSuccess)                    \
    {                                        \
        CUABORT(cudaGetErrorString(err));    \
    }                                        \
}
//-----------------------------------------------------------------------------
inline void cuPrintFileLine(const char *pFile, int nLine) {
    std::cout << "\"" << pFile << "\"(" << nLine << ")";
}
//-----------------------------------------------------------------------------
inline void cuPrintError(const char *pMsg, const char *pFile, int nLine) {
    std::cerr << "An error occured at ";
    cuPrintFileLine(pFile, nLine);
    std::cerr << ": " << pMsg << std::endl;
}
//***************************************************************************//
//**  Device Kernal Functions												 //
//***************************************************************************//
//-----------------------------------------------------------------------------
// TODO: Comments for DIST_CMP
struct DIST_CMP {
    __host__ __device__ bool operator()(
        const CDatabase::DIST &d1,
        const CDatabase::DIST &d2) const {
        return d1.dist < d2.dist;
    }
};
//-----------------------------------------------------------------------------
// TODO: Comments for cuDistances
__global__ void cuDistances(const float *pDatabase,
                            const float *pItem,
                            CDatabase::DIST *pResults,
                            int64_t nItemLen,
                            int64_t nBaseIdx) {
    int64_t iItem = nBaseIdx + blockIdx.x * blockDim.x + threadIdx.x;
    pResults[iItem].dist = 0.0f;
    for (int iElem = 0; iElem < nItemLen; ++iElem) {
        float fDiff = pItem[iElem] - pDatabase[iItem * nItemLen + iElem];
        pResults[iItem].dist += fDiff * fDiff;
    }
}
//-----------------------------------------------------------------------------
__global__ void cuGetSamples(const CDatabase::DIST *pResults,
                             int64_t nSampleInterval,
                             CDatabase::DIST *pSamples,
                             int64_t nBaseIdx) {
    int64_t iDstItem = nBaseIdx + blockIdx.x * blockDim.x + threadIdx.x;
    int64_t iSrcItem = iDstItem * nSampleInterval;
    pSamples[iDstItem].id = pResults[iSrcItem].id;
    pSamples[iDstItem].dist = pResults[iSrcItem].dist;
}
//-----------------------------------------------------------------------------
template<typename _Ty>
_Ty *cuMalloc0(int64_t nUnitCnt) {
    _Ty *pDevMem = 0;
    CUCHECK(cudaMalloc((void **) &pDevMem, sizeof(_Ty) * nUnitCnt));
    CUCHECK(cudaMemset((void *) pDevMem, 0, sizeof(_Ty) * nUnitCnt));
    return pDevMem;
}
//***************************************************************************//
//**  CDatabase Implementation												 //
//***************************************************************************//
//-----------------------------------------------------------------------------
// Constructor
CDatabase::CDatabase()
    : m_nCapacity(0), m_nItemLen(0), m_nCuThreads(0), m_nCuBlocks(0) {
    int32_t nGpuCnt = GetGpuCount();
    CUASSERT(nGpuCnt > 0);
    m_ItemCnts.resize(nGpuCnt, 0);

    for (int32_t i = 0; i < nGpuCnt; ++i) {
        cudaDeviceProp devProp;
        CUCHECK(cudaGetDeviceProperties(&devProp, i));
        if (m_nCuThreads < devProp.maxThreadsPerBlock || m_nCuThreads == 0) {
            const_cast<int32_t &>(m_nCuThreads) = devProp.maxThreadsPerBlock;
        }
        if (m_nCuBlocks < devProp.maxGridSize[2] || m_nCuBlocks == 0) {
            const_cast<int32_t &>(m_nCuBlocks) = devProp.maxGridSize[2];
        }
    }
}
//-----------------------------------------------------------------------------
// Deconstructor
CDatabase::~CDatabase() {
    Clear();
}
//-----------------------------------------------------------------------------
// Get total number of installed GPUs
int32_t CDatabase::GetGpuCount() const {
    std::lock_guard<std::mutex> locker(const_cast<std::mutex &>(m_Mutex));
    int32_t nGpuCnt = 0;
    CUCHECK(cudaGetDeviceCount(&nGpuCnt));
    return nGpuCnt;
}
//-----------------------------------------------------------------------------
// Get item length
int32_t CDatabase::GetItemLength() const {
    std::lock_guard<std::mutex> locker(const_cast<std::mutex &>(m_Mutex));
    return m_nItemLen;
}
//-----------------------------------------------------------------------------
// Get total number of items added to all GPUs
int64_t CDatabase::GetTotalItems() const {
    std::lock_guard<std::mutex> locker(const_cast<std::mutex &>(m_Mutex));
    int64_t nTotalItems = 0;
    for (auto &nGpuItemCnt : m_ItemCnts) {
        if (nGpuItemCnt >= 0) {
            nTotalItems += nGpuItemCnt;
        }
    }
    return nTotalItems;
}
//-----------------------------------------------------------------------------
// TODO: Comment for Initialize
bool CDatabase::Initialize(int64_t nCapacity, int32_t nItemLen,
                           int32_t nGpuMask) {
    std::lock_guard<std::mutex> locker(m_Mutex);
    // Checking Parameters
    if (m_nCapacity != 0) {
        return false;
    }
    CUASSERT(nCapacity > 0);
    CUASSERT(nItemLen > 0);
    int32_t nGpuCnt = (int32_t) m_ItemCnts.size();

    if (nGpuMask < 0) {
        std::fill(m_ItemCnts.begin(), m_ItemCnts.end(), 0);
    }
    else {
        CUASSERT((nGpuMask >> nGpuCnt) == 0);

        for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu) {
            if (nGpuMask & (1 << iGpu)) {
                m_ItemCnts[iGpu] = 0;
            }
            else {
                m_ItemCnts[iGpu] = -1;
            }
        }
    }
    m_ItemSets.resize(nGpuCnt, 0);
    m_QueryItem.resize(nGpuCnt, 0);
    m_QueryResults.resize(nGpuCnt, 0);
    for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu) {
        if (_UseGpu(iGpu)) {
            m_ItemSets[iGpu] = cuMalloc0<float>(nCapacity * nItemLen);
            m_QueryItem[iGpu] = cuMalloc0<float>(nItemLen);
            m_QueryResults[iGpu] = cuMalloc0<DIST>(nCapacity);
        }
    }
    m_nCapacity = nCapacity;
    m_nItemLen = nItemLen;
    return true;
}
//-----------------------------------------------------------------------------
// TODO: Comment for Initialize
void CDatabase::Clear() {
    std::lock_guard<std::mutex> locker(m_Mutex);
    if (m_nCapacity > 0) {
        CUCHECK(cudaDeviceSynchronize());
        for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
            if (_UseGpu(iGpu)) {
                CUCHECK(cudaFree(m_ItemSets[iGpu]));
                CUCHECK(cudaFree(m_QueryItem[iGpu]));
                CUCHECK(cudaFree(m_QueryResults[iGpu]));
            }
        }
        m_ItemSets.clear();
        m_QueryItem.clear();
        m_QueryResults.clear();
        m_nCapacity = 0;
        m_nItemLen = 0;
    }
}
//-----------------------------------------------------------------------------
// TODO: Comment for ResetItems
void CDatabase::ResetItems() {
    std::lock_guard<std::mutex> locker(m_Mutex);
    CUCHECK(cudaDeviceSynchronize());
    for (auto &iGpuItemCnt : m_ItemCnts) {
        if (iGpuItemCnt > 0) {
            iGpuItemCnt = 0;
        }
    }
}
//-----------------------------------------------------------------------------
// TODO: Comment for AddItems
void CDatabase::AddItems(const float *pBatchItems, const int64_t *pBatchIds,
                         int64_t nBatchItemCnt) {
    std::lock_guard<std::mutex> locker(m_Mutex);
    CUCHECK(cudaDeviceSynchronize());
    //Checking Parameter list
    if (nBatchItemCnt == 0) {
        return;
    }
    CUASSERT(pBatchItems != 0);
    CUASSERT(pBatchIds != 0);
    std::vector<int64_t> gpuAddItems(m_ItemCnts);
    // Determin the number of items add for each gpu
    {
        int64_t nUsedGpuCnt = 0;
        int64_t nTotalCap = 0;
        int64_t nTotalItems = 0;
        for (auto &nGpuItemCnt : m_ItemCnts) {
            if (nGpuItemCnt >= 0) {
                ++nUsedGpuCnt;
                nTotalItems += nGpuItemCnt;
                nTotalCap += m_nCapacity;
            }
        }
        CUASSERT(nTotalItems + nBatchItemCnt <= nTotalCap);
        for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
            if (m_ItemCnts[iGpu] >= 0) {
                gpuAddItems[iGpu] += nBatchItemCnt / nUsedGpuCnt;
            }
            else {
                gpuAddItems[iGpu] = nTotalCap;
            }
        }
        for (int64_t iRem = 0; iRem < nBatchItemCnt % nUsedGpuCnt; ++iRem) {
            int32_t iMinGpu = std::min_element(
                gpuAddItems.begin(),
                gpuAddItems.end()
            ) - gpuAddItems.begin();
            ++gpuAddItems[iMinGpu];
        }
        for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
            if (m_ItemCnts[iGpu] >= 0) {
                gpuAddItems[iGpu] -= m_ItemCnts[iGpu];
            }
        }
    }
    int64_t nInputBaseIdx = 0;
    std::vector<DIST> resBuf;
    for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
        if (_UseGpu(iGpu)) {
            CUCHECK(cudaMemcpyAsync(
                m_ItemSets[iGpu] + m_ItemCnts[iGpu] * m_nItemLen,
                pBatchItems + nInputBaseIdx * m_nItemLen,
                gpuAddItems[iGpu] * m_nItemLen * sizeof(float), //In bytes
                cudaMemcpyHostToDevice
            ));

            resBuf.resize(gpuAddItems[iGpu]);
            for (int64_t iItem = 0; iItem < (int64_t) resBuf.size(); ++iItem) {
                resBuf[iItem].id = pBatchIds[nInputBaseIdx + iItem];
            }

            CUCHECK(cudaMemcpyAsync(
                m_QueryResults[iGpu] + m_ItemCnts[iGpu],
                resBuf.data(),
                gpuAddItems[iGpu] * sizeof(DIST),
                cudaMemcpyHostToDevice
            ));

            nInputBaseIdx += gpuAddItems[iGpu];
            m_ItemCnts[iGpu] += gpuAddItems[iGpu];
        }
    }
}
//-----------------------------------------------------------------------------
// TODO: Comment for NearestN
void CDatabase::NearestN(const float *pItem, int64_t N, DIST *pResults) {
    std::lock_guard<std::mutex> locker(m_Mutex);
    CUCHECK(cudaDeviceSynchronize());
    CUASSERT(m_nCapacity > 0);
    // Checking parameters
    CUASSERT(pItem != 0);
    CUASSERT(N > 0);
    CUASSERT(pResults != 0);
    int64_t nTotalItems = 0;
    for (auto &nGpuItemCnt : m_ItemCnts) {
        if (nGpuItemCnt >= 0) {
            nTotalItems += nGpuItemCnt;
        }
    }
    CUASSERT(N <= nTotalItems);
    _UploadQueryItem(pItem);
    _DoQuery();
    float fMaxDist = std::numeric_limits<float>::max();
    int64_t nSamples = (int64_t) std::sqrt((float) nTotalItems * N) * 2.0f;
    int64_t nSampleInterval = nTotalItems / nSamples;
    if (nSampleInterval > 2) {
        fMaxDist = _SampleMaxDist(N, nSampleInterval);
    }
    std::vector<DIST> results(nTotalItems);
    _DownloadResults(fMaxDist, results);
    // Sorting results
    thrust::device_vector<DIST> devBuf(results.begin(), results.end());
    thrust::sort(devBuf.begin(), devBuf.end(), DIST_CMP());
    devBuf.resize(N);
    thrust::copy(devBuf.begin(), devBuf.end(), pResults);
}
//-----------------------------------------------------------------------------
// TODO: Comment for _UploadQueryItem
void CDatabase::_UploadQueryItem(const float *pItem) {
    for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
        if (_UseGpu(iGpu)) {
            CUCHECK(cudaMemcpyAsync(
                m_QueryItem[iGpu],
                pItem,
                sizeof(float) * m_nItemLen,
                cudaMemcpyHostToDevice
            ));
        }
    }
}
//-----------------------------------------------------------------------------
// TODO: Comment for _DoQuery
void CDatabase::_DoQuery() {
    CUCHECK(cudaDeviceSynchronize());
    int64_t nItemsPerQuery = m_nCuBlocks * m_nCuThreads;
    for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
        if (_UseGpu(iGpu)) {
            int64_t nQueryCnt = m_ItemCnts[iGpu] / nItemsPerQuery;
            for (int64_t iQuery = 0; iQuery < nQueryCnt; ++iQuery) {
                cuDistances << < m_nCuBlocks, m_nCuThreads >> > (
                    m_ItemSets[iGpu],
                        m_QueryItem[iGpu],
                        m_QueryResults[iGpu],
                        m_nItemLen,
                        iQuery * nItemsPerQuery
                );
            }
            int64_t nRemainBlocks = (m_ItemCnts[iGpu] % nItemsPerQuery)
                / m_nCuThreads;
            if (nRemainBlocks > 0) {
                cuDistances << < nRemainBlocks, m_nCuThreads >> > (
                    m_ItemSets[iGpu],
                        m_QueryItem[iGpu],
                        m_QueryResults[iGpu],
                        m_nItemLen,
                        nQueryCnt * nItemsPerQuery
                );
            }
            int64_t nItemsRemains = m_ItemCnts[iGpu] % m_nCuThreads;
            if (nItemsRemains > 0) {
                cuDistances << < 1, nItemsRemains >> > (
                    m_ItemSets[iGpu],
                        m_QueryItem[iGpu],
                        m_QueryResults[iGpu],
                        m_nItemLen,
                        m_ItemCnts[iGpu] - nItemsRemains
                );
            }
        }
    }
}
//-----------------------------------------------------------------------------
// TODO: Comment for _SampleMaxDist
float CDatabase::_SampleMaxDist(int64_t N, int64_t nSampleInterval) {
    // Alloc buffer for store samples
    std::vector<thrust::device_vector<DIST>> samples(m_ItemCnts.size());
    for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
        if (_UseGpu(iGpu)) {
            int64_t nSamples = m_ItemCnts[iGpu] / nSampleInterval;
            if (nSamples >= m_nCuBlocks * m_nCuThreads) {
                return std::numeric_limits<float>::max();
            }
            samples[iGpu].resize(nSamples);
        }
    }
    CUCHECK(cudaDeviceSynchronize());
    // Retrieve samples
    for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
        if (_UseGpu(iGpu)) {
            DIST *pSamples = thrust::raw_pointer_cast(samples[iGpu].data());
            int32_t nCudaBlks = (int32_t) samples[iGpu].size() / m_nCuThreads;
            cuGetSamples << < nCudaBlks, m_nCuThreads >> > (
                m_QueryResults[iGpu],
                    nSampleInterval,
                    pSamples,
                    0
            );
            int32_t nRemains = (int32_t) (samples[iGpu].size() % m_nCuThreads);
            if (nRemains > 0) {
                cuGetSamples << < 1, nRemains >> > (
                    m_QueryResults[iGpu],
                        nSampleInterval,
                        pSamples,
                        samples[iGpu].size() - nRemains
                );
            }
        }
    }
    CUCHECK(cudaDeviceSynchronize());
    thrust::host_vector<DIST> maxN;
    thrust::host_vector<DIST> merged;
    for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
        if (_UseGpu(iGpu)) {
            auto &gpuSmp = samples[iGpu];
            thrust::sort(gpuSmp.begin(), gpuSmp.end(), DIST_CMP());

            int64_t nTop = std::min((int64_t) gpuSmp.size(), N);
            gpuSmp.resize(nTop);
            thrust::host_vector<DIST> src(gpuSmp.begin(), gpuSmp.end());
            merged.resize(maxN.size() + src.size());
            thrust::merge(
                maxN.begin(),
                maxN.end(),
                src.begin(),
                src.end(),
                merged.begin(),
                DIST_CMP()
            );
            merged.resize(N);
            merged.swap(maxN);
        }
    }
    return maxN[N - 1].dist;
}
//-----------------------------------------------------------------------------
void CDatabase::_DownloadResults(float fMaxDist, std::vector<DIST> &results) {
    CUCHECK(cudaDeviceSynchronize());
    int64_t nCopied = 0;
    for (int32_t iGpu = 0; iGpu < (int32_t) m_ItemCnts.size(); ++iGpu) {
        if (_UseGpu(iGpu)) {
            auto iSrcBeg = thrust::device_pointer_cast(m_QueryResults[iGpu]);
            auto iSrcEnd = iSrcBeg + m_ItemCnts[iGpu];
            thrust::copy(iSrcBeg, iSrcEnd, results.data() + nCopied);
            auto iDstBeg = results.begin() + nCopied;
            auto iNewEnd = std::remove_if(
                iDstBeg,
                iDstBeg + m_ItemCnts[iGpu],
                [&fMaxDist](const DIST &d1) -> bool {
                    return d1.dist > fMaxDist;
                });

            nCopied += (iNewEnd - iDstBeg);
        }
    }
    results.resize(nCopied);
}
//-----------------------------------------------------------------------------
bool CDatabase::_UseGpu(int32_t iGpu) {
    if (m_ItemCnts[iGpu] >= 0) {
        CUCHECK(cudaSetDevice(iGpu));
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
void LoadDatabaseFromFile(const std::string &strFile, CDatabase &db) {
    const int64_t nItemsPerBatch = 1024 * 1024;
    std::ifstream dataFile(strFile, std::ios::out | std::ios::binary);
    CUASSERT(dataFile.is_open());
    int64_t nItemCnt;
    CUASSERT(dataFile.read((char *) &nItemCnt, sizeof(nItemCnt)));
    int32_t nItemLen;
    CUASSERT(dataFile.read((char *) &nItemLen, sizeof(nItemLen)));
    db.Initialize(nItemCnt, nItemLen);
    int64_t nBatchCnt = nItemCnt / nItemsPerBatch;
    int64_t nFloatsPerBatch = nItemLen * nItemsPerBatch;
    std::vector<float> itemsBuf(nFloatsPerBatch);
    std::vector<int64_t> idsBuf(nItemsPerBatch);
    for (int64_t iBatch = 0; iBatch < nBatchCnt; ++iBatch) {
        std::cout << "\"" << __FILE__ << "\"(" << __LINE__ << ") ";
        std::cout << "Loading Batch " << iBatch;
        std::cout << "/" << nBatchCnt << std::endl;
        for (int64_t iItem = 0; iItem < nItemsPerBatch; ++iItem) {
            CUASSERT(dataFile.read(
                (char *) (idsBuf.data() + iItem),
                sizeof(int64_t)
            ));
            CUASSERT(dataFile.read(
                (char *) (itemsBuf.data() + iItem * nItemLen),
                sizeof(float) * nItemLen
            ));
        }
        db.AddItems(itemsBuf.data(), idsBuf.data(), nItemsPerBatch);
    }
    std::cout << "\"" << __FILE__ << "\"(" << __LINE__ << ") ";
    std::cout << "Loading Batch " << nBatchCnt;
    std::cout << "/" << nBatchCnt << std::endl;
    int64_t nRemains = nItemCnt % nItemsPerBatch;
    if (nRemains > 0) {
        for (int64_t iItem = 0; iItem < nRemains; ++iItem) {
            CUASSERT(dataFile.read(
                (char *) (idsBuf.data() + iItem),
                sizeof(int64_t)
            ));
            CUASSERT(dataFile.read(
                (char *) (itemsBuf.data() + iItem * nItemLen),
                sizeof(float) * nItemLen
            ));
        }
        db.AddItems(itemsBuf.data(), idsBuf.data(), nRemains);
    }
}

}
//===========================================================================//
