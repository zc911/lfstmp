#include <string>
#include <memory>
#include <iostream>
#include <grpc++/grpc++.h>
#include "service/witness_service.h"
#include "service/witness_service_asyn.h"

using namespace std;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using namespace dg;

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cout << "Usage: " << argv[0] << " [S|A] PORT" << endl;
        return 0;
    }

    char addr[1024];
    sprintf(addr, "0.0.0.0:%s", argv[2]);
    std::string server_address(addr);

    bool asyn = false;
    if (string(argv[1]) == "A") {
        asyn = true;
    }

    if (asyn) {

        WitnessServiceAsynImpl service(addr);
        std::cout << "Server(Asyn) listening on " << server_address
                  << std::endl;
        service.Run();

    } else {
        WitnessServiceImpl service;

        ServerBuilder builder;
        builder.AddListeningPort(server_address,
                                 grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        std::unique_ptr<Server> server(builder.BuildAndStart());
        std::cout << "Server listening on " << server_address << std::endl;
        server->Wait();
    }

    return 0;
}

