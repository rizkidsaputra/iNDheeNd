// Prototype aplikasi sink laboratorium. Tidak dijalankan sebelum pilot disetujui.
#include <ndn-cxx/face.hpp>
#include <ndn-cxx/name.hpp>
#include <boost/json.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <time.h>

static long long monotonicNs() {
  timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<long long>(t.tv_sec)*1000000000LL+t.tv_nsec;
}
static bool allowed(const std::string& p) {
  const std::string base="/poc11c/attack/unavailable/";
  return p.rfind(base,0)==0 && p.size()>base.size();
}
int main(int argc,char** argv) {
  if(argc==2 && std::string(argv[1])=="--self-test") {
    if(!allowed("/poc11c/attack/unavailable/unit") || allowed("/demo/producer/p1")) return 1;
    ndn::Name n("/poc11c/attack/unavailable/unit");
    std::cout<<"SELF_TEST_OK: parsing/guard; tidak ada Face atau registrasi prefix\n";
    return n.empty()?1:0;
  }
  if(argc!=3 || std::string(argv[1])!="--execute-after-pilot-approval" || !allowed(argv[2])) {
    std::cerr<<"Memerlukan persetujuan pilot; gunakan --self-test untuk pengujian offline\n";
    return 2;
  }
  const char* transport=std::getenv("NDN_CLIENT_TRANSPORT");
  if(!transport || std::string(transport)!="unix:///run/nfd/p1.sock") {
    std::cerr<<"Sink harus berada di namespace p1 dengan transport p1.sock\n";return 3;
  }
  try {
    ndn::Face face;
    face.setInterestFilter(ndn::Name(argv[2]),
      [](const auto&,const ndn::Interest& interest) {
        boost::json::object o{{"event","sink_interest_received"},{"monotonic_ns",monotonicNs()},
                              {"name",interest.getName().toUri()},{"data_generated",false}};
        std::cout<<boost::json::serialize(o)<<std::endl;
        // Sengaja tanpa Data/NACK: prefix tetap terdaftar ke aplikasi lokal.
      },
      [](const auto&) {std::cout<<"{\"event\":\"sink_prefix_registered\"}"<<std::endl;},
      [](const auto&,const std::string& reason) {std::cerr<<reason<<std::endl;std::exit(4);});
    face.processEvents();
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 5;}
}
