#include "rad_isa_capture.h"
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>
#include <sys/wait.h>
int main(int argc,char** argv) {
    if (argc < 2) return 20;
    std::ifstream f(argv[1],std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)),{});
    if (argc > 2 && std::string(argv[2]) == "hash") {
        std::puts(rad_isa_capture::sha256(bytes.data(),bytes.size()).c_str());return 0;
    }
    if (bytes.empty()) return 21;
    if (!rad_isa_capture::object(bytes.data(),bytes.size(),0)) return 2;
    if (argc > 2 && std::string(argv[2]) == "fork") {
        auto pid=fork();
        if (pid==0) _exit(rad_isa_capture::resolved(bytes.data(),bytes.size(),0,"smoke")?0:2);
        int status=0;waitpid(pid,&status,0);return WIFEXITED(status)?WEXITSTATUS(status):3;
    }
    if (argc > 2 && std::string(argv[2]) == "threads") {
        std::vector<std::thread> ts;
        for(int n=0;n<8;++n)ts.emplace_back([&]{
            if(!rad_isa_capture::resolved(bytes.data(),bytes.size(),0,"smoke"))std::abort();
        });
        for(auto& t:ts)t.join();
    } else if (!rad_isa_capture::resolved(bytes.data(),bytes.size(),0,"smoke")) return 2;
    return 0;
}
