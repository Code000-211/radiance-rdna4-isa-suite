// Capture real libref invocation bytes without a model, server or fabricated engine shim.
#include "capture.h"
#include <iostream>
int main(int argc,char** argv){
 if(argc!=3)return 2;
 try{
  setenv("RAD_REPLAY_CAPTURE_DIR",argv[2],1);
  rad::Registry reg;if(reg.load_plugin(argv[1],0)!=RAD_OK)return 2;
  rad::Geometry g;g.set_i("M",1);g.set_i("n",4);g.set_s("dtype","f32");
  rad::Registry::RowFilter filter=[](const rad::KernelRow& r,const rad::Geometry&){return std::string(r.info->name)=="ref_add"?"":"other";};
  auto* row=reg.select_in("libref","add",g,RAD_DOMAIN_HOST,&filter,nullptr);if(!row)return 2;
  float x[4]={1,2,3,4},y[4]={4,3,2,1},o[4]={};RadTensor t[3]{};
  for(int i=0;i<3;++i){t[i].data=i==0?x:i==1?y:o;t[i].dtype=RAD_F32;t[i].rank=2;t[i].shape[0]=1;t[i].shape[1]=4;t[i].stride[0]=4;t[i].stride[1]=1;}
  RadArgs a{t,3,g.params(),g.n_params(),nullptr,0,nullptr,0,1};
  rad_replay::Capture capture(reg,row,a,reg.schema("add"),{16,16,16},nullptr,"kbench-seeded");
  if(!capture.active())return 2;
  if(row->info->launch(&a,nullptr)!=RAD_OK)return 1;
  capture.finish();std::cout<<"captured real libref add\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
