import os
from pathlib import Path
import subprocess
import sys

from scripts.run_frozen_final_stage import source_environment


def test_controller_child_imports_its_checkout_instead_of_old_environment(tmp_path):
    checkout = tmp_path / 'current'
    old = tmp_path / 'old'
    for root, value in ((checkout, 'current'), (old, 'old')):
        (root / 'graphnav').mkdir(parents=True)
        (root / 'graphnav/__init__.py').write_text('')
        (root / 'graphnav/checkpointing.py').write_text(f'MARKER={value!r}\n')
    (checkout / 'scripts').mkdir()
    entry = checkout / 'scripts/freeze.py'
    entry.write_text('from graphnav.checkpointing import MARKER\nprint(MARKER)\n')
    inherited = dict(os.environ, PYTHONPATH=str(old), CUSTOM_KEEP='preserved')
    assert subprocess.check_output([sys.executable, str(entry)], env=inherited, text=True).strip() == 'old'
    env = source_environment(checkout, inherited)
    assert env['CUSTOM_KEEP'] == 'preserved' and inherited['PYTHONPATH'] == str(old)
    assert Path(env['PYTHONPATH']) == checkout.resolve()
    assert subprocess.check_output([sys.executable, str(entry)], env=env, text=True).strip() == 'current'
