//*****************************************************************************
// A great source code just likes a poem that any comment shall profanes it.
//														-- devymex@gmail.com
//*****************************************************************************


#ifndef DATABASE_H_
#define DATABASE_H_

#include <mutex>
#include <vector>

namespace dg {
class CDatabase
{
 public:
    struct DIST
    {
        int64_t	id;			// Item index of one device
        float	dist;		// Squared distance
    };
 protected:
    const int32_t			m_nCuThreads;
    const int32_t			m_nCuBlocks;
    int64_t					m_nCapacity;	// Capacity per gpu
    int32_t					m_nItemLen;		// Item length;
    std::vector<int64_t>	m_ItemCnts;		// Current items
    // Pointers of Device Memory
    std::vector<float*>		m_ItemSets;
    std::vector<float*> 	m_QueryItem;
    std::vector<DIST*>		m_QueryResults;
    std::mutex				m_Mutex;
 public:
    // Constructor
    CDatabase();
    // Deconstructor
    virtual ~CDatabase();
    // Get total number of installed GPUs
    int32_t	GetGpuCount() const;
    // Get item length
    int32_t	GetItemLength() const;
    // Get total number of items added to all GPUs
    int64_t	GetTotalItems() const;

    bool RetrieveItemById(int64_t nId, float *pItem);

    bool	Initialize(int64_t nCapacity, int32_t nItemLen,
                       int32_t nGpuMask = -1);
    void	Clear();
    void	ResetItems();
    void	AddItems(const float *pItems, const int64_t *pIds, int64_t nCnt);
    void	NearestN(const float *pItem, int64_t N, DIST *pResults);
 protected:
    void	_UploadQueryItem(const float *pItem);
    void	_DoQuery();
    float	_SampleMaxDist(int64_t N, int64_t nSampleInterval);
    void	_DownloadResults(float fMaxDist, std::vector<DIST> &results);
    bool	_UseGpu(int32_t iGpu);
};
void LoadDatabaseFromFile(const std::string &strFile, CDatabase &db);
}
#endif /* DATABASE_H_ */
