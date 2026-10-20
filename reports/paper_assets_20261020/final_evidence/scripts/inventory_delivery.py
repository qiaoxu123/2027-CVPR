"""Hash the public extension; private RGB never enters the delivery inventory."""
from common import *


if __name__=='__main__':
    output=FINAL/'DELIVERY_INVENTORY.json'
    files={str(p.relative_to(FINAL)):sha(p) for p in sorted(FINAL.rglob('*'))
        if p.is_file() and p!=output and 'local_only' not in p.parts and '__pycache__' not in p.parts}
    dump(output,{'source_experiment_commit':'16452896e8b169a9c540ce6de7f7d00b17b453a7',
        'private_RGB_included':False,'public_files_sha256':files,
        'note':'Commit SHA is reported after Git commit; this file does not attempt a self-referential commit hash.'})
    print('Public files inventoried:',len(files))
