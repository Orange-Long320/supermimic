#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

// Point-sampled gyro trapezoid and linearly varying world acceleration.
// No truth pose, online leveling or zero-velocity correction.
struct PointIntegrator {
  using V = std::array<double,3>;
  std::array<double,4> q{1,0,0,0}, x{};
  V bias{}, gyro{}, previous{}, world{};
  int64_t stamp=0;
  uint64_t samples=0;
  double max_gap=0;
  bool initialized=false;
  static V rotate(const std::array<double,4>& q,const V& v) {
    V t{2*(q[2]*v[2]-q[3]*v[1]),2*(q[3]*v[0]-q[1]*v[2]),2*(q[1]*v[1]-q[2]*v[0])};
    return {v[0]+q[0]*t[0]+q[2]*t[2]-q[3]*t[1],
            v[1]+q[0]*t[1]+q[3]*t[0]-q[1]*t[2],
            v[2]+q[0]*t[2]+q[1]*t[1]-q[2]*t[0]};
  }
  void initialize(const V& b,const std::array<double,4>& quat) {
    if(initialized)throw std::runtime_error("No in-run calibration reset");
    double norm=0;for(double v:quat){if(!std::isfinite(v))throw std::runtime_error("Invalid calibration");norm+=v*v;}
    if(!(norm>0))throw std::runtime_error("Zero calibration quaternion");
    for(double v:b)if(!std::isfinite(v))throw std::runtime_error("Invalid bias");
    bias=b;q=quat;for(double& v:q)v/=std::sqrt(norm);initialized=true;
  }
  void update(int64_t t,const V& raw,const V& acc) {
    if(!initialized || t<=0)throw std::runtime_error("Invalid initialization/timestamp");
    for(int i=0;i<3;++i){if(!std::isfinite(raw[i])||!std::isfinite(acc[i]))throw std::runtime_error("Invalid IMU");gyro[i]=raw[i]-bias[i];}
    if(samples) {
      double dt=(t-stamp)*1e-9;
      if(!(dt>0 && dt<=.040))throw std::runtime_error("IMU duplicate/rollback or gap above 40 ms");
      if(dt>max_gap)max_gap=dt;
      V rv;double theta2=0;for(int i=0;i<3;++i){rv[i]=.5*(previous[i]+gyro[i])*dt;theta2+=rv[i]*rv[i];}
      double theta=std::sqrt(theta2),s=theta<1e-8 ? .5-theta2/48 : std::sin(theta/2)/theta;
      std::array<double,4> d{std::cos(theta/2),s*rv[0],s*rv[1],s*rv[2]},n;
      n={q[0]*d[0]-q[1]*d[1]-q[2]*d[2]-q[3]*d[3],
         q[0]*d[1]+q[1]*d[0]+q[2]*d[3]-q[3]*d[2],
         q[0]*d[2]-q[1]*d[3]+q[2]*d[0]+q[3]*d[1],
         q[0]*d[3]+q[1]*d[2]-q[2]*d[1]+q[3]*d[0]};
      double norm=0;for(double v:n)norm+=v*v;for(int i=0;i<4;++i)q[i]=n[i]/std::sqrt(norm);
      V a=rotate(q,acc);a[2]-=9.81;
      for(int i=0;i<2;++i){x[i]+=x[i+2]*dt+(2*world[i]+a[i])*dt*dt/6;x[i+2]+=.5*(world[i]+a[i])*dt;}
      world=a;
    } else {world=rotate(q,acc);world[2]-=9.81;}
    previous=gyro;stamp=t;++samples;
  }
};
