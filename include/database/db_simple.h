#ifndef _dgfacesdk_database_simple_h_
#define _dgfacesdk_database_simple_h_
#include <string>
#include <database.h>
#include "common.h"
namespace DGFace{

class SimpleDatabase : public Database {
    public:
        SimpleDatabase(const std::string& db_path, Verification* verifier);
        virtual ~SimpleDatabase(void);
    protected:
        virtual void load();
        virtual void save();
    private:
        std::string _db_path;
};
}
#endif

