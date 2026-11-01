import unittest
from supermimic.deploy.x2_command_groups import JOINT_GROUPS, group_command


class GroupCommandTests(unittest.TestCase):
    def test_name_mapping_not_input_order(self):
        names=list(reversed([x for group in JOINT_GROUPS.values() for x in group]))
        packet={'names':names,**{k:list(range(31)) for k in ('position','velocity','effort','stiffness','damping')}}
        groups=group_command(packet)
        for group,items in groups.items():
            self.assertEqual([x['name'] for x in items],list(JOINT_GROUPS[group]))
            for row in items:self.assertEqual(row['position'],names.index(row['name']))

    def test_missing_duplicate_and_short_fields_rejected(self):
        names=[x for group in JOINT_GROUPS.values() for x in group]
        packet={'names':names,**{k:[0.]*31 for k in ('position','velocity','effort','stiffness','damping')}}
        packet['names']=names[:-1]+names[:1]
        with self.assertRaises(ValueError):group_command(packet)
        packet['names']=names;packet['effort']=[]
        with self.assertRaises(ValueError):group_command(packet)


if __name__=='__main__':unittest.main()
