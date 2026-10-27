import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from schema import validate_public, NETWORK_KEYS
from test_model import inputs


class SchemaTests(unittest.TestCase):
    def test_private_fields_rejected(self):
        p=dict(network=inputs(),spatial={})
        validate_public(p)
        p['network']['target_xy']=torch.ones(2)
        with self.assertRaises(ValueError):validate_public(p)
    def test_nonfinite_rejected(self):
        p=dict(network=inputs(),spatial={});p['network']['pair'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):validate_public(p)

if __name__=='__main__':unittest.main()
