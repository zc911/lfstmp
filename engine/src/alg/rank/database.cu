#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "database.h"

namespace dg {
//***************************************************************************//
//**  Error Handling Macros and Function									 //
//***************************************************************************//

//---------------------------------------------------------------------------
// TODO: Comments for CUABORT
#define CUABORT(msg)                        \
{                                            \
    cuPrintError(msg, __FILE__, __LINE__);    \
    exit(-1);                                \
}

//---------------------------------------------------------------------------
// TODO: Comments for CUASSERT
#define CUASSERT(exp)                        \
{                                            \
    if (!(exp))                                \
    {                                        \
        CUABORT(#exp);                        \
    }                                        \
}

//---------------------------------------------------------------------------
// TODO: Comments for CUCHECK
#define CUCHECK(exp)                        \
{                                            \
    cudaError_t err = (exp);                \
    if (err != cudaSuccess)                    \
    {                                        \
        CUABORT(cudaGetErrorString(err));    \
    }                                        \
}

//---------------------------------------------------------------------------
// TODO: Comments for cuPrintError
inline void cuPrintError(const char *pMsg, const char *pFile, int nLine) {
    std::cerr << "An error occured at \"" << pFile << "\"(" << nLine;
    std::cerr << "): " << pMsg << std::endl;
}



//***************************************************************************//
//**  Device Kernal Functions												 //
//***************************************************************************//

//---------------------------------------------------------------------------
// TODO: Comments for DIST_CMP
struct DIST_CMP {
    __host__ __device__ bool operator()(
        const CDatabase::DIST &d1,
        const CDatabase::DIST &d2) const {
        return d1.dist < d2.dist;
    }
};

//---------------------------------------------------------------------------
// TODO: Comments for DIST_CUT
struct DIST_CUT {
    float *m_pMaxDist;
    __host__ __device__ DIST_CUT(float *pMaxDist)
        : m_pMaxDist(pMaxDist) {
    }
    __host__ __device__ bool operator()(const CDatabase::DIST &d) const {
        return d.dist <= *m_pMaxDist;
    }
};

//---------------------------------------------------------------------------
struct DIST_GATHER {
    CDatabase::DIST *m_pFiltered;
    float m_fCutVal;
    unsigned int *m_pCurIdx;

    __host__ __device__ DIST_GATHER(float fCutVal, CDatabase::DIST *pFiltered,
                                    unsigned int *pCurIdx)
        : m_fCutVal(fCutVal), m_pFiltered(pFiltered), m_pCurIdx(pCurIdx) {
    }

    __device__ void operator()(const CDatabase::DIST &d) const {
        if (d.dist <= m_fCutVal) {
            int idx = atomicInc(m_pCurIdx, 0XFFFFFFFF);
            m_pFiltered[idx].id = d.id;
            m_pFiltered[idx].dist = d.dist;
        }
    }
};

//---------------------------------------------------------------------------
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

//---------------------------------------------------------------------------
__global__ void cuGetSamples(const CDatabase::DIST *pResults,
                             int64_t nSampleInterval,
                             CDatabase::DIST *pSamples) {
    int64_t iDstItem = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t iSrcItem = iDstItem * nSampleInterval;
    pSamples[iDstItem].id = pResults[iSrcItem].id;
    pSamples[iDstItem].dist = pResults[iSrcItem].dist;
}

//***************************************************************************//
//**  Helper Functions														 //
//***************************************************************************//

//---------------------------------------------------------------------------
template<typename _Ty, typename _Cmp>
void gpuSort(std::vector<_Ty> &ary, _Cmp cmp) {
    thrust::device_vector<_Ty> devBuf;
    try {
        devBuf.resize(ary.size());
    }
    catch (...) {
        CUABORT("Out of Memory!");
    }
    thrust::copy(ary.begin(), ary.end(), devBuf.begin());
    thrust::sort(devBuf.begin(), devBuf.end(), DIST_CMP());
    thrust::copy(devBuf.begin(), devBuf.end(), ary.begin());
}

//---------------------------------------------------------------------------
template<typename _Ty>
_Ty *cuMalloc0(int64_t nUnitCnt) {
    _Ty *pDevMem = 0;
    CUCHECK(cudaMalloc((void **) &pDevMem, sizeof(_Ty) * nUnitCnt));
    CUCHECK(cudaMemset((void *) pDevMem, 0, sizeof(_Ty) * nUnitCnt));
    return pDevMem;
}

//---------------------------------------------------------------------------
// TODO: Comments for cuDistances
template<typename _Ty, typename _Cmp>
void cpuSort(_Ty *data, int64_t len, int grainsize, _Cmp cmp) {
    if (len < grainsize) {
        std::sort(data, data + len, cmp);
    }
    else {
        auto future = std::async(
            cpuSort < _Ty, _Cmp > ,
            data,
            len / 2,
            grainsize,
            cmp
        );
        cpuSort(data + len / 2, len / 2, grainsize, cmp);
        future.wait();
        std::inplace_merge(data, data + len / 2, data + len, cmp);
    }
}


//---------------------------------------------------------------------------
void LoadDatabaseFromFile(const std::string &strFile, CDatabase &db) {
    const int64_t nItemsPerBatch = 1024LL * 1024LL;

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

//***************************************************************************//
//**  CDatabase Implementation												 //
//***************************************************************************//

//---------------------------------------------------------------------------
// Constructor
CDatabase::CDatabase()
    : m_nGPUs(0), m_nItemCnt(0), m_nItemLen(0), m_nPosition(0) {
    CUCHECK(cudaGetDeviceCount(&m_nGPUs));
    CUASSERT(m_nGPUs >= 1);
}

//---------------------------------------------------------------------------
// Deconstructor
CDatabase::~CDatabase() {
    Clear();
}

//---------------------------------------------------------------------------
// TODO: Comment for SetItemCnt
void CDatabase::SetWorkingGPUs(int32_t nGPUs) {
    CUASSERT(m_nItemCnt == 0); //Make sure the database is empty
    CUASSERT(nGPUs > 0);

    int32_t nAvailableGPUs = 0;
    CUCHECK(cudaGetDeviceCount(&nAvailableGPUs));

    CUASSERT(nGPUs <= nAvailableGPUs);
    m_nGPUs = nGPUs;
}

//---------------------------------------------------------------------------
// TODO: Comment for SetItemCnt
void CDatabase::Initialize(int64_t nItemCnt, int64_t nItemLen) {
    CUASSERT(m_nGPUs > 0);        //To make sure the host at least one GPU
    CUASSERT(m_nItemCnt == 0);    //To make sure the database is empty

    // Checking Parameters
    CUASSERT(nItemCnt > 0);
    CUASSERT(nItemCnt % m_nGPUs == 0);
    CUASSERT(nItemLen > 0);

    CUCHECK(cudaDeviceSynchronize());

    int64_t nItemsPerGPU = nItemCnt / m_nGPUs;
    for (int64_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));

        // Create device memory for database on devices
        m_ItemSets.push_back(cuMalloc0<float>(nItemsPerGPU * nItemLen));

        // Create device memory for passing query item to devices
        m_QueryItem.push_back(cuMalloc0<float>(nItemLen));

        // Create device memory for store query results on devices
        m_QueryResults.push_back(cuMalloc0<DIST>(nItemsPerGPU));
    }
    CUCHECK(cudaDeviceSynchronize());

    m_nItemCnt = nItemCnt;
    m_nItemLen = nItemLen;
    m_nPosition = 0;
}

//---------------------------------------------------------------------------
void CDatabase::Clear() {
    CUASSERT(m_nGPUs > 0);        //To make sure the host at least one GPU

    if (m_nItemCnt != 0) {
        for (int64_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
            CUCHECK(cudaSetDevice(iGpu));

            CUCHECK(cudaFree(m_ItemSets[iGpu]));
            m_ItemSets[iGpu] = 0;

            CUCHECK(cudaFree(m_QueryResults[iGpu]));
            m_QueryResults[iGpu] = 0;

            CUCHECK(cudaFree(m_QueryItem[iGpu]));
            m_QueryItem[iGpu] = 0;
        }
    }
    m_ItemSets.clear();
    m_QueryResults.clear();
    m_SampleIdx.clear();

    m_nPosition = 0;
    m_nItemCnt = 0;
    m_nItemLen = 0;
}

//---------------------------------------------------------------------------
// TODO: Comment for AddItems
void CDatabase::AddItems(const float *pBatchItems,
                         const int64_t *pBatchIds,
                         int64_t nBatchItemCnt) {
    CUASSERT(m_nGPUs > 0);        //To make sure the host at least one GPU
    CUASSERT(m_nItemCnt > 0);    //To make sure the database has initialized

    //Checking Parameter list
    CUASSERT(pBatchItems != 0);
    CUASSERT(pBatchIds != 0);

    CUCHECK(cudaDeviceSynchronize());

    int64_t nBatchItemsPerGPU = nBatchItemCnt / m_nGPUs;
    CUASSERT(m_nPosition + nBatchItemsPerGPU <= m_nItemCnt);

    int64_t nBatchFloatsPerGPU = nBatchItemsPerGPU * m_nItemLen;

    // Buffer for initializing item ID in results
    std::vector<DIST> resBuf(nBatchItemsPerGPU);
    for (int64_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));

        CUCHECK(cudaMemcpyAsync(
            m_ItemSets[iGpu] + m_nPosition * m_nItemLen, // offset in floats
            pBatchItems + iGpu * nBatchFloatsPerGPU, // offset in floats
            nBatchFloatsPerGPU * sizeof(float), //In bytes
            cudaMemcpyHostToDevice
        ));

        for (int64_t iItem = 0; iItem < nBatchItemsPerGPU; ++iItem) {
            resBuf[iItem].id = pBatchIds[iGpu * nBatchItemsPerGPU + iItem];
        }
        CUCHECK(cudaMemcpyAsync(
            m_QueryResults[iGpu] + m_nPosition,
            resBuf.data(),
            sizeof(DIST) * nBatchItemsPerGPU,
            cudaMemcpyHostToDevice
        ));
    }
    m_nPosition += nBatchItemsPerGPU;
}

//---------------------------------------------------------------------------
void CDatabase::ResetItems() {
    m_nPosition = 0;
}

//---------------------------------------------------------------------------
void CDatabase::NearestN(const float *pItem, int64_t N, int64_t *pOutIds) {
    CUASSERT(m_nGPUs > 0);        //To make sure the host at least one GPU
    CUASSERT(m_nItemCnt != 0);    //To make sure the database has initialized
    // To make sure the database has filled
    std::cout << m_nPosition << " " << m_nItemCnt << " " << m_nGPUs << std::endl;
    CUASSERT(m_nPosition == m_nItemCnt / m_nGPUs);

    CUASSERT(pItem != 0);
    CUASSERT(N > 0);
    CUASSERT(N < m_nItemCnt / 10);
    CUASSERT(pOutIds != 0);

    _UploadQueryItem(pItem);

    _DoQuery();

    std::vector<DIST> results;
    results.reserve(m_nItemCnt);

    _DownloadResults(results, N);

    gpuSort(results, DIST_CMP());

    for (int64_t i = 0; i < N; ++i) {
        pOutIds[i] = results[i].id;
    }
}

//---------------------------------------------------------------------------
int32_t CDatabase::GetGPUCount() {
    return m_nGPUs;
}

//---------------------------------------------------------------------------
int64_t CDatabase::GetItemCount() {
    return m_nItemCnt;
}

//---------------------------------------------------------------------------
int64_t CDatabase::GetItemLength() {
    return m_nItemLen;
}

//---------------------------------------------------------------------------
void CDatabase::_UploadQueryItem(const float *pItem) {
    CUCHECK(cudaDeviceSynchronize());
    for (int64_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        CUCHECK(cudaMemcpyAsync(
            m_QueryItem[iGpu],
            pItem,
            sizeof(float) * m_nItemLen,
            cudaMemcpyHostToDevice
        ));
    }
}

//---------------------------------------------------------------------------
void CDatabase::GetItem(int iGpu, int iPos, float *pItem, int64_t *pId) {
    CUASSERT(iGpu >= 0 && iGpu < m_nGPUs);
    CUASSERT(iPos >= 0 && iPos < m_nItemCnt / m_nGPUs);
    CUASSERT(pItem != 0);
    CUASSERT(pId != 0);

    CUCHECK(cudaSetDevice(iGpu));
    std::vector<float> item(m_nItemLen);
    CUCHECK(cudaMemcpy(
        pItem,
        m_ItemSets[iGpu] + iPos * m_nItemLen,
        m_nItemLen * sizeof(float),
        cudaMemcpyDeviceToHost)
    );
    DIST dist;
    CUCHECK(cudaMemcpy(
        &dist,
        m_QueryResults[iGpu] + iPos,
        sizeof(dist),
        cudaMemcpyDeviceToHost)
    );
    *pId = dist.id;
}

//---------------------------------------------------------------------------
void CDatabase::_DoQuery() {
    CUCHECK(cudaDeviceSynchronize());

    // m_nItemCnt must can be divided exactly by m_nGPUs
    int64_t nItemsPerGPU = m_nItemCnt / m_nGPUs;
    int64_t nItemsPerQuery = CUDA_BLOCKS * CUDA_THREADS;
    int64_t nQueryCnt = nItemsPerGPU / nItemsPerQuery;
    for (int64_t iQuery = 0; iQuery < nQueryCnt; ++iQuery) {
        _GPUQuery(CUDA_BLOCKS, CUDA_THREADS, iQuery * nItemsPerQuery);
    }

    int64_t nItemsRemains = nItemsPerGPU % nItemsPerQuery;
    if (nItemsRemains > 0) {
        int64_t nRemainBlocks = nItemsRemains / CUDA_THREADS;

        if (nRemainBlocks > 0) {
            _GPUQuery(
                nRemainBlocks,                    // nCudaBlocks
                CUDA_THREADS,                    // nCudaThreads
                nItemsPerGPU - nItemsRemains    // nBaseIdx
            );
            nItemsRemains -= nRemainBlocks * CUDA_THREADS;
        }

        if (nItemsRemains > 0) {
            _GPUQuery(1, nItemsRemains, nItemsPerGPU - nItemsRemains);
        }
    }
    CUCHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------
void CDatabase::_GPUQuery(int64_t nCudaBlocks,
                          int64_t nCudaThreads,
                          int64_t nBaseIdx) {
    CUCHECK(cudaDeviceSynchronize());
    for (int64_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        cuDistances << < nCudaBlocks, nCudaThreads >> > (
            m_ItemSets[iGpu],
                m_QueryItem[iGpu],
                m_QueryResults[iGpu],
                m_nItemLen,
                nBaseIdx
        );
    }
}

//---------------------------------------------------------------------------
void CDatabase::_DownloadResults(std::vector<DIST> &results, int64_t N) {

    CUCHECK(cudaDeviceSynchronize());

    int64_t nItemsPerGPU = m_nItemCnt / m_nGPUs;
    int64_t nSamples = (int64_t) std::sqrt((float) m_nItemCnt * N) * 2.0f;
    nSamples /= CUDA_THREADS * m_nGPUs;
    nSamples *= CUDA_THREADS * m_nGPUs;

    float fMaxDist = _SampleMaxDist(N, nSamples);

    std::vector<int64_t> gatherCnts;
    _CountForGather(fMaxDist, gatherCnts);

    std::vector<thrust::device_vector<DIST>> gatherBufs(m_nGPUs);
    std::vector<unsigned int *> gatherBufIdx(m_nGPUs);
    int nTotalGathered = 0;

    for (int32_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        int nGathered = gatherCnts[iGpu];
        gatherBufs[iGpu].resize(nGathered);
        CUCHECK(cudaMalloc((void **) &gatherBufIdx[iGpu], sizeof(int)));
        CUCHECK(cudaMemset((void *) gatherBufIdx[iGpu], 0, sizeof(int)));
        nTotalGathered += nGathered;
    }
    CUCHECK(cudaDeviceSynchronize());

    int nGathered = 0;
    results.resize(nTotalGathered);
    for (int32_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        auto pResBeg = m_QueryResults[iGpu];
        auto pResEnd = m_QueryResults[iGpu] + nItemsPerGPU;
        DIST *pGatherBuf = thrust::raw_pointer_cast(
            gatherBufs[iGpu].data());
        thrust::for_each(
            thrust::device_pointer_cast(pResBeg),
            thrust::device_pointer_cast(pResEnd),
            DIST_GATHER(fMaxDist, pGatherBuf, gatherBufIdx[iGpu])
        );
        CUCHECK(cudaMemcpy(
            results.data() + nGathered,
            pGatherBuf,
            gatherBufs[iGpu].size() * sizeof(DIST),
            cudaMemcpyDeviceToHost
        ));
        CUCHECK(cudaFree(gatherBufIdx[iGpu]));
        nGathered += gatherBufs[iGpu].size();
    }
    CUCHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------
float CDatabase::_SampleMaxDist(int64_t N, int64_t nSamples) {
    CUASSERT(nSamples > N);
    CUASSERT((nSamples / m_nGPUs) % CUDA_THREADS == 0);
    CUASSERT(nSamples / CUDA_THREADS <= CUDA_BLOCKS);

    CUCHECK(cudaDeviceSynchronize());
    int64_t nSampleInterval = m_nItemCnt / nSamples;
    int64_t nSamplesPerGPU = nSamples / m_nGPUs;
    std::vector<float> maxVals;

    std::vector<thrust::device_vector<DIST>> samples(m_nGPUs);
    for (int32_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        samples[iGpu].resize(nSamplesPerGPU);
    }
    CUCHECK(cudaDeviceSynchronize());
    for (int32_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        DIST *pGpuSamples = thrust::raw_pointer_cast(samples[iGpu].data());
        cuGetSamples << < nSamplesPerGPU / CUDA_THREADS, CUDA_THREADS >> > (
            m_QueryResults[iGpu],
                nSampleInterval,
                pGpuSamples
        );
    }
    CUCHECK(cudaDeviceSynchronize());

    for (int32_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        thrust::sort(samples[iGpu].begin(), samples[iGpu].end(), DIST_CMP());
        DIST maxDist = samples[iGpu][N];
        maxVals.push_back(maxDist.dist);
    }
    float fMaxVal = *std::max_element(maxVals.begin(), maxVals.end());
    CUCHECK(cudaDeviceSynchronize());
    return fMaxVal;
}

//---------------------------------------------------------------------------
void CDatabase::_CountForGather(float fMaxDist,
                                std::vector<int64_t> &gatherCnts) {
    CUCHECK(cudaDeviceSynchronize());
    int64_t nItemsPerGPU = m_nItemCnt / m_nGPUs;
    std::vector<float *> maxDists(m_nGPUs, 0);
    for (int32_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        auto pResBeg = m_QueryResults[iGpu];
        auto pResEnd = m_QueryResults[iGpu] + nItemsPerGPU;
        CUCHECK(cudaMalloc((void **) &maxDists[iGpu], sizeof(float)));
        CUCHECK(cudaMemcpy(
            maxDists[iGpu],
            &fMaxDist,
            sizeof(fMaxDist),
            cudaMemcpyHostToDevice
        ));
        int64_t nGathered = thrust::count_if(
            thrust::device_pointer_cast(pResBeg),
            thrust::device_pointer_cast(pResEnd),
            DIST_CUT(maxDists[iGpu])
        );
        gatherCnts.push_back(nGathered);
    }
    CUCHECK(cudaDeviceSynchronize());
    for (int32_t iGpu = 0; iGpu < m_nGPUs; ++iGpu) {
        CUCHECK(cudaSetDevice(iGpu));
        CUCHECK(cudaFree(maxDists[iGpu]));
    }
}
}
//===========================================================================//
