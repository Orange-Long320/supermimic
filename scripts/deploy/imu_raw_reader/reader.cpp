// Raw sensor transport only. No attitude fusion, kinematics or motor interfaces.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <array>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h>
#include <sys/prctl.h>
#include <signal.h>

constexpr uint64_t MAGIC = 0x5832524157303032ULL;
struct Frame {
  uint64_t magic = MAGIC;
  int64_t stamp, received;
  std::array<double, 6> values;
  std::array<char, 64> frame{};
  std::array<double,31> q{}, qd{};
  std::array<int64_t,4> joint_stamps{}, joint_received{};
  std::array<int32_t,4> domains{};
  uint32_t groups = 0, joints = 0;
};
struct Joints {
  int64_t stamp, received;
  int32_t domain;
  uint32_t mask = 0;
  std::array<double,31> q{}, qd{};
};
static_assert(sizeof(Frame) == 720, "Raw sensor wire layout mismatch");
static int64_t wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}
int main(int argc, char **argv) {
  try {
    if (::prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || ::getppid() == 1 || argc < 2) return 1;
    uint16_t endian = 1;
    if (*reinterpret_cast<uint8_t *>(&endian) != 1) throw std::runtime_error("Little endian required");
    std::unordered_map<std::string,int> indices;
    std::istringstream names(argv[1]);std::string name;
    while (std::getline(names,name,',')) {
      int index=indices.size();
      if (!indices.emplace(name,index).second) throw std::runtime_error("Duplicate profile joint");
    }
    if (indices.size()!=31) throw std::runtime_error("Expected 31 profile joints");
    rclcpp::init(argc, argv);
    auto options = rclcpp::NodeOptions().enable_rosout(false).start_parameter_services(false)
      .start_parameter_event_publisher(false);
    auto node = std::make_shared<rclcpp::Node>("supermimic_raw_pelvis_sensor_reader", options);
    std::array<std::deque<Joints>,4> history;
    std::array<rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr,4> joints;
    const char *groups[]={"leg","waist","head","arm"};
    for (int g=0;g<4;++g) {
      joints[g]=node->create_subscription<aimdk_msgs::msg::JointStateArray>(
        std::string("/aima/hal/joint/")+groups[g]+"/state",rclcpp::QoS(256).best_effort().durability_volatile(),
        [&,g](aimdk_msgs::msg::JointStateArray::ConstSharedPtr m) {
          Joints row;row.stamp=static_cast<int64_t>(m->header.stamp.sec)*1000000000LL+m->header.stamp.nanosec;
          row.received=wall_ns();row.domain=m->state.value;
          if(row.stamp<=0 || (!history[g].empty() && row.stamp<=history[g].back().stamp))
            throw std::runtime_error("Joint timestamp duplicate/rollback");
          for(const auto &j:m->joints) {
            auto it=indices.find(j.name);if(it==indices.end())continue;
            int i=it->second;uint32_t bit=1u<<i;
            if((row.mask&bit) || !std::isfinite(j.position) || !std::isfinite(j.velocity))
              throw std::runtime_error("Invalid encoder value or duplicate joint");
            row.mask|=bit;row.q[i]=j.position;row.qd[i]=j.velocity;
          }
          history[g].push_back(row);if(history[g].size()>128)history[g].pop_front();
        });
    }
    auto imu = node->create_subscription<sensor_msgs::msg::Imu>("/aima/hal/imu/torso/state",
      rclcpp::QoS(256).best_effort().durability_volatile(),
      [&](sensor_msgs::msg::Imu::ConstSharedPtr m) {
        if (m->header.frame_id.size() >= 64) throw std::runtime_error("IMU frame name too long");
        Frame f;
        f.stamp = static_cast<int64_t>(m->header.stamp.sec)*1000000000LL+m->header.stamp.nanosec;
        f.received = wall_ns();
        f.values = {m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z,
                    m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z};
        std::memcpy(f.frame.data(), m->header.frame_id.data(), m->header.frame_id.size());
        for(int g=0;g<4;++g) for(auto it=history[g].rbegin();it!=history[g].rend();++it) {
          if(it->stamp>f.stamp)continue;
          if(f.stamp-it->stamp>25000000LL)break;
          f.groups|=1u<<g;f.joint_stamps[g]=it->stamp;f.joint_received[g]=it->received;f.domains[g]=it->domain;
          f.joints|=it->mask;
          for(int i=0;i<31;++i)if(it->mask&(1u<<i)){f.q[i]=it->q[i];f.qd[i]=it->qd[i];}
          break;
        }
        ssize_t n;
        do { n = ::write(STDOUT_FILENO, &f, sizeof(f)); } while (n < 0 && errno == EINTR);
        if (n != sizeof(f)) throw std::runtime_error("Raw sensor pipe write failed");
      });
    rclcpp::spin(node);
    rclcpp::shutdown();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
