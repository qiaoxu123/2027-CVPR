import subprocess
import unittest
from pathlib import Path


class BundleTests(unittest.TestCase):
    def test_detector_data_package_is_present_and_not_ignored(self):
        root=Path(__file__).resolve().parents[1]
        rel='runtime/waldo_vendor/ultralytics/data/__init__.py'
        self.assertTrue((root/rel).is_file(),'Bundled Ultralytics data package is required')
        if (root/'.git').exists():
            p=subprocess.run(['git','check-ignore','--no-index','--quiet',rel],cwd=root)
            self.assertEqual(p.returncode,1,'Root cache ignore must not exclude vendored data modules')

if __name__=='__main__':unittest.main()
