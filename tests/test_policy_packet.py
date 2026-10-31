import copy
import unittest
from supermimic.deploy.policy_packet import validate_policy_packet


class PolicyPacketTests(unittest.TestCase):
    def setUp(self):
        self.names = tuple('joint_'+str(i) for i in range(31))
        self.packet = dict(schema=1, session_id='one', joint_names=list(self.names), position=[0., 0., .65],
            quat_wxyz=[1., 0., 0., 0.], angular_velocity_body=[0.]*3,
            joint_position=[0.]*31, joint_velocity=[0.]*31, joint_stamps=[9.988]*4,
            joint_received_wall=[9.991]*4, state_stamp=9.99, published_wall=9.995,
            joint_domains=[0]*4, motor_output_enabled=False)

    def test_valid_causal_packet(self):
        self.assertIs(validate_policy_packet(self.packet, self.names, 10.), self.packet)

    def test_future_joint_not_hidden_by_fresh_receipt(self):
        self.packet['joint_stamps'][2] = 9.992
        with self.assertRaisesRegex(ValueError, 'future joint'):
            validate_policy_packet(self.packet, self.names, 10.)

    def test_retained_packet_stales_despite_new_receive(self):
        self.packet['published_wall'] = 10.2
        with self.assertRaisesRegex(ValueError, 'stale'):
            validate_policy_packet(self.packet, self.names, 10.21)

    def test_hardware_domain_and_order_fail_closed(self):
        for key, value in [('joint_domains', [0, 1, 0, 0]), ('joint_names', list(reversed(self.names)))]:
            packet = copy.deepcopy(self.packet); packet[key] = value
            with self.assertRaises(ValueError): validate_policy_packet(packet, self.names, 10.)

    def test_bad_values_and_partial_packet(self):
        for key, value in [('joint_position', [float('nan')]*31), ('quat_wxyz', [2., 0., 0., 0.]),
                           ('joint_velocity', [0.]*30), ('joint_stamps', [9.95, 9.988, 9.988, 9.988])]:
            packet = copy.deepcopy(self.packet); packet[key] = value
            with self.assertRaises(ValueError): validate_policy_packet(packet, self.names, 10.)


if __name__ == '__main__': unittest.main()
