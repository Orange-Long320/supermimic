"""Contract-only self check: run the real publisher gate without ROS or motors."""
import sys
import time
from pathlib import Path

trial = Path(sys.argv[1])
sys.path.insert(0, str(trial))
from x2_hardware_publisher import validated_contract  # noqa: E402

plan = validated_contract(trial / 'contract-dryrun/plan.json',
                          trial / 'contract-dryrun/consent.json', time.time())
print('gate accepted')
print('safety_mode', plan['safety_mode'])
print('manual_supervision', plan['manual_supervision'])
print('output_hz', plan['output_hz'], 'max_tick_gap', plan['max_tick_gap'])
print('max_target_step', plan['max_target_step'][0])
print('command bounds checked per joint:', len(plan['command_lower']))
