// Bounded sensor subscriptions only. Retains actual header, RMW metadata and
// callback time in memory; writes once after capture. No actuator interfaces.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <filesystem>

using Imu = sensor_msgs::msg::Imu;
struct Row {
  int64_t header, callback, source, received;
  uint64_t publication, reception;
  std::array<uint8_t, RMW_GID_STORAGE_SIZE> gid;
};
static int64_t wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}
class Audit : public rclcpp::Node {
 public:
  Audit() : Node("supermimic_readonly_imu_transport_audit") {
    for (int i=0;i<3;++i) {
      rows[i].reserve(65000);
      auto qos=rclcpp::QoS(400).durability_volatile();
      if (i==0) qos.best_effort(); else qos.reliable();
      const std::string topic=i==2 ? "/supermimic/x2_imu_relay/imu" : "/aima/hal/sensor/lidar_chest_front/imu";
      subs[i]=create_subscription<Imu>(topic,qos,
        [this,i](Imu::ConstSharedPtr msg,const rclcpp::MessageInfo &info) {
          if(rows[i].size()>=65000) throw std::runtime_error("Audit memory capacity exceeded");
          const auto &m=info.get_rmw_message_info();
          Row row{static_cast<int64_t>(msg->header.stamp.sec)*1000000000LL+msg->header.stamp.nanosec,
            wall_ns(),static_cast<int64_t>(m.source_timestamp),static_cast<int64_t>(m.received_timestamp),
            m.publication_sequence_number,m.reception_sequence_number,{}};
          std::copy(std::begin(m.publisher_gid.data),std::end(m.publisher_gid.data),row.gid.begin());
          rows[i].push_back(row);
        });
    }
  }
  std::array<std::vector<Row>,3> rows;
 private:
  std::array<rclcpp::Subscription<Imu>::SharedPtr,3> subs;
};
int main(int argc,char **argv) {
  if(argc!=3) return 2;
  const double seconds=std::stod(argv[1]); const std::string out=argv[2];
  if(!std::isfinite(seconds)||seconds<5||seconds>300||std::filesystem::exists(out)) return 2;
  rclcpp::init(argc,argv);
  auto node=std::make_shared<Audit>();
  rclcpp::executors::SingleThreadedExecutor executor;executor.add_node(node);
  const auto until=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
  while(rclcpp::ok()&&std::chrono::steady_clock::now()<until) executor.spin_once(std::chrono::milliseconds(10));
  std::ofstream file(out); if(!file) throw std::runtime_error("Cannot create audit output");
  file<<"reader,header_ns,callback_ns,rmw_source_ns,rmw_received_ns,publication_sequence,reception_sequence,gid\n";
  const char *labels[]={"native_best_effort","native_reliable","relay_reliable"};
  for(size_t i=0;i<3;++i) for(const auto &r:node->rows[i]) {
    file<<labels[i]<<','<<r.header<<','<<r.callback<<','<<r.source<<','<<r.received<<','
      <<r.publication<<','<<r.reception<<',';
    for(auto v:r.gid) file<<std::hex<<std::setfill('0')<<std::setw(2)<<static_cast<unsigned>(v);
    file<<std::dec<<'\n';
  }
  file.close();
  std::cout<<"captured "<<node->rows[0].size()<<' '<<node->rows[1].size()<<' '<<node->rows[2].size()<<std::endl;
  rclcpp::shutdown();return 0;
}
