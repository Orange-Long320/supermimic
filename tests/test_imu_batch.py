import copy
import json
import unittest
from supermimic.deploy.imu_batch import ImuBatchDecoder


class ImuBatchTest(unittest.TestCase):
    def setUp(self):
        self.d = ImuBatchDecoder('a'*32)
        self.p = {'schema':1,'session':'a'*32,'sequence':1,'published_ns':1000,'rejected':0,'overflow':0,
                  'samples':[[900,950,1.,2.,3.,4.,5.,6.],[901,951,2.,3.,4.,5.,6.,7.]]}
    def read(self, p): return self.d.decode(json.dumps(p), 1010)
    def test_preserves_original_samples_and_empty_batches(self):
        self.assertEqual(self.read(self.p), self.p['samples'])
        p=copy.deepcopy(self.p); p['sequence']=2; p['samples']=[]
        self.assertEqual(self.read(p), [])
    def test_rejects_gaps_and_recovers(self):
        self.read(self.p); p=copy.deepcopy(self.p); p['sequence']=3
        with self.assertRaises(ValueError): self.read(p)
        p['sequence']=4; p['samples']=[]
        self.assertEqual(self.read(p), [])
    def test_invalid_packet_fails(self):
        for error in ('reordered','future','nan','session','duplicate'):
            p=copy.deepcopy(self.p)
            if error=='reordered': p['samples'].reverse()
            if error=='future': p['samples'][0][1]=1001
            if error=='nan': p['samples'][0][2]=float('nan')
            if error=='session': p['session']='b'*32
            if error=='duplicate': p['samples'][1][0]=900
            with self.subTest(error=error), self.assertRaises(ValueError): self.read(p)
        with self.assertRaises(ValueError): self.d.decode(json.dumps(self.p), 50_001_001)
    def test_dropped_sample_and_overflow_fail_closed(self):
        self.read(self.p)
        p=copy.deepcopy(self.p); p['sequence']=2; p['rejected']=1; p['samples']=[]
        with self.assertRaises(ValueError): self.read(p)
        p['sequence']=3; p['overflow']=1
        with self.assertRaises(ValueError): self.read(p)

    def test_replay_recovers_dropped_packet_using_original_samples(self):
        d=ImuBatchDecoder('a'*32)
        row=[1,900,950,1.,2.,3.,4.,5.,6.]
        p={'schema':2,'session':'a'*32,'sequence':1,'published_ns':1000,'rejected':0,'source_seq':1,'samples':[row]}
        self.assertEqual(d.decode(json.dumps(p),1010), [row[1:]])
        r2=[2,902,952,2.,3.,4.,5.,6.,7.]; r3=[3,904,954,3.,4.,5.,6.,7.,8.]
        p.update(sequence=3,source_seq=3,samples=[row,r2,r3])
        self.assertEqual(d.decode(json.dumps(p),1010), [r2[1:],r3[1:]])
        self.assertEqual(d.recovered_packets,1)
        p.update(sequence=4)
        self.assertEqual(d.decode(json.dumps(p),1010), [])

    def test_replay_unrecoverable_gap_and_source_error_reset(self):
        d=ImuBatchDecoder('a'*32)
        row=[1,900,950,1.,2.,3.,4.,5.,6.]
        p={'schema':2,'session':'a'*32,'sequence':1,'published_ns':1000,'rejected':0,'source_seq':1,'samples':[row]}
        d.decode(json.dumps(p),1010)
        p.update(sequence=3,source_seq=3,samples=[[3,904,954,3.,4.,5.,6.,7.,8.]])
        with self.assertRaises(ValueError): d.decode(json.dumps(p),1010)
        p.update(sequence=4,source_seq=4,samples=[[3,904,954,3.,4.,5.,6.,7.,8.], [4,905,955,3.,4.,5.,6.,7.,8.]])
        self.assertEqual(d.decode(json.dumps(p),1010), [p['samples'][-1][1:]])
        p.update(sequence=5,rejected=1)
        with self.assertRaises(ValueError): d.decode(json.dumps(p),1010)
        p.update(sequence=6)
        self.assertEqual(d.decode(json.dumps(p),1010), [])


if __name__=='__main__': unittest.main()
