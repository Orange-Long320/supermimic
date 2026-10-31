// Diagnostic only: retain the latest real sample per group, never command motors.
#include <rclcpp/rclcpp.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

using JointArray = aimdk_msgs::msg::JointStateArray;
using namespace std::chrono_literals;

class Relay : public rclcpp::Node {
 public:
  explicit Relay(const std::string &session) : Node("supermimic_joint_relay"), session_(session) {
    const std::array<std::string, 4> groups = {"leg", "waist", "head", "arm"};
    pub_ = create_publisher<std_msgs::msg::String>("/supermimic/x2_joint_relay/state", rclcpp::QoS(5).best_effort());
    for (size_t i=0; i<groups.size(); ++i) {
      subs_[i] = create_subscription<JointArray>("/aima/hal/joint/"+groups[i]+"/state", rclcpp::QoS(1).best_effort(),
        [this,i](JointArray::ConstSharedPtr m) {
          rows_[i] = m; received_[i] = now_ns(); ++counts_[i];
        });
    }
    timer_ = create_wall_timer(10ms, [this,groups]() {
      std::ostringstream s; s << std::setprecision(17);
      s << "{\"schema\":1,\"session\":\"" << session_ << "\",\"sequence\":" << sequence_++
        << ",\"published_ns\":" << now_ns() << ",\"samples\":[";
      bool first = true;
      for (size_t i=0; i<groups.size(); ++i) {
        if (!rows_[i] || counts_[i] == sent_[i]) continue;
        const auto &m=*rows_[i]; bool valid=true;
        for (const auto &j:m.joints) {
          if (j.name.empty() || j.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos || !std::isfinite(j.position) || !std::isfinite(j.velocity)) valid=false;
        }
        if (!valid) { sent_[i]=counts_[i]; continue; }
        if (!first) s << ','; first=false;
        s << "{\"group\":\"" << groups[i] << "\",\"header_ns\":"
          << (static_cast<int64_t>(m.header.stamp.sec)*1000000000LL+m.header.stamp.nanosec)
          << ",\"received_ns\":" << received_[i] << ",\"domain\":" << static_cast<int>(m.state.value)
          << ",\"source_count\":" << counts_[i] << ",\"joints\":[";
        bool fj=true;
        for (const auto &j:m.joints) {
          if (!fj) s << ','; fj=false;
          s << "[\"" << j.name << "\"," << j.position << ',' << j.velocity << ',' << j.error_code << ']';
        }
        s << "]}"; sent_[i]=counts_[i];
      }
      s << "]}"; std_msgs::msg::String out; out.data=s.str(); pub_->publish(out);
    });
  }
 private:
  static int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
  std::string session_; uint64_t sequence_=0;
  std::array<JointArray::ConstSharedPtr,4> rows_{};
  std::array<int64_t,4> received_{}; std::array<uint64_t,4> counts_{},sent_{};
  std::array<rclcpp::Subscription<JointArray>::SharedPtr,4> subs_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_; rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc,char **argv) {
  if (argc != 2) return 2;
  std::string session=argv[1];
  if (session.size()!=32 || session.find_first_not_of("0123456789abcdef")!=std::string::npos) return 2;
  rclcpp::init(argc,argv); rclcpp::spin(std::make_shared<Relay>(session)); rclcpp::shutdown();
}
