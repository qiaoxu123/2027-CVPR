from v5core import *
def run():
 RESULT.mkdir(exist_ok=True)
 files=list(V4.glob('*.py'))+[V4/'results'/f for f in ['relations.jsonl.gz','public_inputs.jsonl.gz','frozen_frames.jsonl.gz','road_fields.jsonl.gz','geometry_checks.jsonl.gz','summary.json','freeze_manifest.json']]
 writej(RESULT/'v4_lock.json',{'base_commit':'6ae8b68d4f73eaabbbb7aeb077169d0d7565adb4','files':{str(f.relative_to(V4)):sha(f) for f in files},'protocol_sha256':sha(HERE/'PROTOCOL.md'),'split':'TRAIN ONLY','external_validation_read':False,'new_GT_opened':False})
 print('v4 source/results locked',len(files))
if __name__=='__main__':run()
