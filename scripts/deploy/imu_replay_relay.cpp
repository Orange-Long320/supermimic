// Diagnostic sensor relay only: forward validated native IMU and batch its
// actual received samples. No actuator publishers or robot control clients.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std::chrono_literals;
using Imu = sensor_msgs::msg::Imu;
struct Sample { Imu::ConstSharedPtr msg; int64_t received_ns; uint64_t source_seq; };

class Relay : public rclcpp::Node {
 public:
  explicit Relay(const std::string &session) : Node("supermimic_imu_batch_relay"), session_(session) {
    imu_pub_ = create_publisher<Imu>("/supermimic/x2_imu_relay/imu", rclcpp::QoS(400));
    batch_pub_ = create_publisher<std_msgs::msg::String>("/supermimic/x2_imu_relay/batch", rclcpp::QoS(10).best_effort());
    sub_ = create_subscription<Imu>("/aima/hal/sensor/lidar_chest_front/imu", rclcpp::QoS(50).best_effort(),
      [this](Imu::ConstSharedPtr m) {
        const auto received = now_ns();
        const int64_t stamp = static_cast<int64_t>(m->header.stamp.sec)*1000000000LL+m->header.stamp.nanosec;
        const auto &w=m->angular_velocity; const auto &a=m->linear_acceleration;
        if (m->header.frame_id != "lidar_imu_chest_front" || stamp <= last_stamp_ || stamp > received ||
            received-stamp > 100000000LL || !std::isfinite(w.x) || !std::isfinite(w.y) || !std::isfinite(w.z) ||
            !std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.z)) { ++rejected_; return; }
        last_stamp_=stamp;
        imu_pub_->publish(*m);
        if (queue_.size() == 16) queue_.pop_front();
        queue_.push_back({m,received,++source_seq_});
      });
    timer_ = create_wall_timer(10ms, [this]() {
      std::ostringstream s; s << std::setprecision(17);
      s << "{\"schema\":2,\"session\":\"" << session_ << "\",\"sequence\":" << sequence_++
        << ",\"published_ns\":" << now_ns() << ",\"rejected\":" << rejected_
        << ",\"source_seq\":" << source_seq_ << ",\"samples\":[";
      bool first=true;
      for (const auto &row:queue_) {
        const auto &m=*row.msg; const auto &w=m.angular_velocity; const auto &a=m.linear_acceleration;
        if (!first) s << ','; first=false;
        s << '[' << row.source_seq << ',' << (static_cast<int64_t>(m.header.stamp.sec)*1000000000LL+m.header.stamp.nanosec)
          << ',' << row.received_ns << ',' << w.x << ',' << w.y << ',' << w.z << ',' << a.x << ',' << a.y << ',' << a.z << ']';
      }
      s << "]}";
      std_msgs::msg::String out; out.data=s.str(); batch_pub_->publish(out);
    });
  }
 private:
  static int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
  std::string session_; uint64_t sequence_=0,rejected_=0,source_seq_=0; int64_t last_stamp_=0;
  std::deque<Sample> queue_;
  rclcpp::Subscription<Imu>::SharedPtr sub_; rclcpp::Publisher<Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr batch_pub_; rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc,char **argv) {
  if (argc != 2) return 2;
  std::string session=argv[1];
  if (session.size()!=32 || session.find_first_not_of("0123456789abcdef")!=std::string::npos) return 2;
  rclcpp::init(argc,argv); rclcpp::spin(std::make_shared<Relay>(session)); rclcpp::shutdown();
}
