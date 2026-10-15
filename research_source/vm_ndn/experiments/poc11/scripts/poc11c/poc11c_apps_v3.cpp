// Pilot laboratorium PoC11C: producer, consumer, reporter dan collector.
#include <ndn-cxx/face.hpp>
#include <ndn-cxx/security/key-chain.hpp>
#include <ndn-cxx/security/signing-helpers.hpp>
#include <ndn-cxx/util/scheduler.hpp>
#include <boost/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <functional>
#include <cmath>
#include <time.h>
static int64_t now(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return int64_t(t.tv_sec)*1000000000+t.tv_nsec;}
static void log(boost::json::object o){o["monotonic_ns"]=now();std::cout<<boost::json::serialize(o)<<std::endl;}
int main(int argc,char**argv){
  if(argc<8){std::cerr<<"role node run start_ns duration_s parameter prefix_or_directory\n";return 2;}
  std::string role=argv[1],node=argv[2],run=argv[3],arg=argv[7];
  int64_t start=std::stoll(argv[4]);double duration=std::stod(argv[5]),param=std::stod(argv[6]);
  try{
    ndn::Face face;ndn::Scheduler scheduler(face.getIoContext());ndn::KeyChain keys;
    uint64_t sequence=0;std::function<void()> emit;
    std::function<void(ndn::Interest,int64_t)> serve;
    auto at=[&](int64_t target,std::function<void()> f){scheduler.schedule(ndn::time::nanoseconds(std::max<int64_t>(0,target-now())),std::move(f));};
    scheduler.schedule(ndn::time::nanoseconds(std::max<int64_t>(0,start+int64_t((duration+6)*1e9)-now())),[&]{face.shutdown();});
    auto registrationOk=[&](const auto&){log({{"event","prefix_registered"},{"role",role},{"node",node},{"run_id",run}});};
    auto registrationError=[&](const auto&,const std::string& reason){log({{"event","registration_error"},{"reason",reason}});face.shutdown();};
    if(role=="producer" || role=="reporter"){
      std::string prefix=role=="producer"?arg:"/monitor/"+node+"/telemetry/poc11c/"+run;
      serve=[&](ndn::Interest interest,int64_t expires){
        std::string content;
        if(role=="producer") content=std::string(1024,'D');
        else{
          auto seq=interest.getName().at(-1).toUri();
          if(seq.find_first_not_of("0123456789")!=std::string::npos)return;
          std::ifstream in(arg+"/"+node+"/"+seq+".json");
          if(!in){
            if(now()<expires)scheduler.schedule(ndn::time::milliseconds(25),[&,interest,expires]{serve(interest,expires);});
            else log({{"event","payload_unavailable"},{"name",interest.getName().toUri()}});
            return;
          }
          std::stringstream buffer;buffer<<in.rdbuf();content=buffer.str();
        }
        ndn::Data data(interest.getName());data.setFreshnessPeriod(ndn::time::milliseconds(1000));
        data.setContent(std::string_view(content));keys.sign(data,ndn::security::signingWithSha256());face.put(data);
        log({{"event","data_sent"},{"node",node},{"run_id",run},{"name",interest.getName().toUri()},{"content_bytes",content.size()}});
      };
      face.setInterestFilter(ndn::Name(prefix),[&](const auto&,const ndn::Interest& interest){serve(interest,now()+1000000000LL);},registrationOk,registrationError);
    }else if(role=="consumer"){
      emit=[&]{
        int64_t target=start+int64_t(sequence*1e9/param);
        if(target>=start+int64_t(duration*1e9))return;
        ndn::Name name(arg);name.append(run).append(node).append(std::to_string(sequence));
        ndn::Interest interest(name);interest.setCanBePrefix(false).setMustBeFresh(true).setInterestLifetime(ndn::time::milliseconds(node[0]=='a'?4000:2000));
        log({{"event","interest_submitted"},{"run_id",run},{"node",node},{"name",name.toUri()},{"sequence",sequence},{"target_ns",target},{"lifetime_ms",node[0]=='a'?4000:2000}});
        face.expressInterest(interest,[&,n=name.toUri()](const auto&,const ndn::Data& data){
          auto& content=data.getContent();bool good=content.value_size()==1024;
          for(size_t i=0;i<content.value_size()&&good;++i)good=content.value()[i]=='D';
          log({{"event","data_received"},{"node",node},{"run_id",run},{"name",n},{"content_bytes",content.value_size()},{"valid_content",good},{"exact_name",data.getName().toUri()==n}});
        },[&,n=name.toUri()](const auto&,const ndn::lp::Nack& nack){log({{"event","nack"},{"node",node},{"name",n},{"reason",int(nack.getReason())}});},
        [&,n=name.toUri()](const auto&){log({{"event","timeout"},{"node",node},{"name",n}});});
        ++sequence;at(start+int64_t(sequence*1e9/param),emit);
      };at(start,emit);
    }else if(role=="collector"){
      int count=int(duration/3);std::stringstream list(arg);std::string router;
      while(std::getline(list,router,','))for(int seq=1;seq<=count;++seq){
        int idx=std::stoi(router.substr(1))-1;
        int64_t sourceTarget=start+(seq-1)*3000000000LL+idx*100000000LL;
        at(sourceTarget+300000000LL,[&,router,seq,sourceTarget]{
          ndn::Name name("/monitor/"+router+"/telemetry/poc11c/"+run+"/"+std::to_string(seq));
          ndn::Interest interest(name);interest.setCanBePrefix(false).setMustBeFresh(true).setInterestLifetime(ndn::time::milliseconds(3600));
          log({{"event","fetch_submitted"},{"router_id",router},{"sequence",seq},{"target_ns",sourceTarget},{"deadline_ns",sourceTarget+1900000000LL},{"name",name.toUri()}});
          face.expressInterest(interest,[&,router,seq](const auto&,const ndn::Data& data){
            auto& c=data.getContent();std::string raw(reinterpret_cast<const char*>(c.value()),c.value_size());
            log({{"event","telemetry_received"},{"router_id",router},{"sequence",seq},{"raw_payload",raw},{"data_name",data.getName().toUri()}});
          },[&,router,seq](const auto&,const ndn::lp::Nack& nack){log({{"event","telemetry_nack"},{"router_id",router},{"sequence",seq},{"reason",int(nack.getReason())}});},
          [&,router,seq](const auto&){log({{"event","telemetry_timeout"},{"router_id",router},{"sequence",seq}});});
        });
      }
    }else return 3;
    log({{"event","app_ready"},{"role",role},{"node",node},{"run_id",run}});face.processEvents();
  }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 4;}
}
