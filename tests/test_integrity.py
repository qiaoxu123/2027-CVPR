import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import common
import prepare


class IntegrityTests(unittest.TestCase):
    def test_waldo_and_annotation_replacement_rejected(self):
        c=dict(checkpoint_sha256='checkpoint',source_sha256={},waldo_sha256='waldo',
            annotation_path='/tmp/local-refinement-annotation.json',annotation_sha256='annotation')
        waldo=common.WALDO/'WALDO30_yolov8l-p2_1024x1024.pt'
        def digest(path):
            return {str(common.CHECKPOINT):'checkpoint',str(waldo):'wrong',c['annotation_path']:'annotation'}[str(path)]
        with patch.object(common,'sha',side_effect=digest):
            with self.assertRaises(ValueError):common.verify_sources(c)
        def annotation_changed(path):
            return {str(common.CHECKPOINT):'checkpoint',str(waldo):'waldo',c['annotation_path']:'wrong'}[str(path)]
        with patch.object(common,'sha',side_effect=annotation_changed):
            with self.assertRaises(ValueError):common.verify_sources(c)

    def test_partial_cache_replays_preceding_history(self):
        calls=[]
        def compute(step):calls.append(step);return f'computed-{step}'
        got=list(prepare.causal_extract([1,2,3],{1:'cached-1',3:'cached-3'},compute))
        self.assertEqual(calls,[1,2]);self.assertEqual(got,[(1,'cached-1'),(2,'computed-2'),(3,'cached-3')])
        calls.clear();list(prepare.causal_extract([1,2],{1:'cached-1',2:'cached-2'},compute))
        self.assertEqual(calls,[])
    def test_old_label_checkpoint_cannot_calibrate_on_new_labels(self):
        with self.assertRaises(ValueError):common.validate_cache_signature({'cache_signature':{'labels':'old'}},{'labels':'new'})
        common.validate_cache_signature({'cache_signature':{'labels':'same'}},{'labels':'same'})
    def test_relocated_cached_rows_and_source_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data').mkdir();p=root/'data'/'train_1_2.pt';p.touch()
            with patch.object(common,'ROOT',root):
                self.assertEqual(common.resolve_row_path('/old/machine/data/train_1_2.pt'),p)
        self.assertEqual(common.resolve_source_path('/missing/GraphNav/graphnav/test_missing.py'),common.SOURCE/'graphnav/test_missing.py')
    def test_source_verification_follows_actual_runtime_not_old_location(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'GraphNav/graphnav/model.py';old.parent.mkdir(parents=True);old.write_text('old')
            bundle=Path(d)/'runtime/GraphNav'
            with patch.object(common,'SOURCE',bundle):
                self.assertEqual(common.resolve_source_path(old),bundle/'graphnav/model.py')

if __name__=='__main__':unittest.main()
