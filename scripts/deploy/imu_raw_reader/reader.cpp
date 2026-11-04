// Raw sensor transport only. No attitude fusion, kinematics or motor interfaces.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <sys/prctl.h>
#include <signal.h>

struct Frame {
  uint64_t magic = 0x5832494D55353030ULL;
  int64_t stamp, received;
  std::array<double, 6> values;
  std::array<char, 64> frame{};
};
static_assert(sizeof(Frame) == 136, "Raw IMU wire layout mismatch");
int main(int argc, char **argv) {
  try {
    if (::prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || ::getppid() == 1) return 1;
    uint16_t endian = 1;
    if (*reinterpret_cast<uint8_t *>(&endian) != 1) throw std::runtime_error("Little endian required");
    rclcpp::init(argc, argv);
    auto options = rclcpp::NodeOptions().enable_rosout(false).start_parameter_services(false)
      .start_parameter_event_publisher(false);
    auto node = std::make_shared<rclcpp::Node>("supermimic_raw_pelvis_imu_reader", options);
    auto sub = node->create_subscription<sensor_msgs::msg::Imu>("/aima/hal/imu/torso/state",
      rclcpp::QoS(256).best_effort().durability_volatile(),
      [](sensor_msgs::msg::Imu::ConstSharedPtr m) {
        if (m->header.frame_id.size() >= 64) throw std::runtime_error("IMU frame name too long");
        Frame f;
        f.stamp = static_cast<int64_t>(m->header.stamp.sec)*1000000000LL+m->header.stamp.nanosec;
        f.received = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
        f.values = {m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z,
                    m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z};
        std::memcpy(f.frame.data(), m->header.frame_id.data(), m->header.frame_id.size());
        ssize_t n;
        do { n = ::write(STDOUT_FILENO, &f, sizeof(f)); } while (n < 0 && errno == EINTR);
        if (n != sizeof(f)) throw std::runtime_error("Raw IMU pipe write failed");
      });
    rclcpp::spin(node);
    rclcpp::shutdown();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
