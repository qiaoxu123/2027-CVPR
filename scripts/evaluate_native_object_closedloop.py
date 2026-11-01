"""Isolated native-object navigation entry point; preserve original controller/protocol."""
import argparse,json,sys
from pathlib import Path
from graphnav.checkpointing import atomic_json,file_sha256


def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--native-checkpoint',required=True);p.add_argument('--native-control',choices=['real','zero','shuffle','no_relation'],default='real');p.add_argument('--native-frozen-manifest');p.add_argument('--native-history-window',type=int,default=1);a,rest=p.parse_known_args()
 def value(flag):
  for i,item in enumerate(rest):
   if item==flag:return rest[i+1]
   if item.startswith(flag+'='):return item.split('=',1)[1]
  return None
 if value('--model') not in (None,'graphnav-v1'):raise ValueError('Native object adapter requires graphnav-v1')
 if value('--model') is None:rest+=['--model','graphnav-v1']
 split=value('--split');checkpoint=Path(a.native_checkpoint)
 if not (checkpoint.parent/'complete.json').exists():raise ValueError('Completed native training required')
 if split in ('val_unseen','test_unseen'):
  if not a.native_frozen_manifest:raise ValueError('Held-out native evaluation requires Seen-selected frozen native manifest')
  frozen=json.loads(Path(a.native_frozen_manifest).read_text())
  if frozen.get('native_checkpoint_sha')!=file_sha256(checkpoint) or not frozen.get('seen_validation_complete'):raise ValueError('Unfrozen native checkpoint')
  if frozen.get('native_history_window',1)!=a.native_history_window or frozen.get('native_control','real')!=a.native_control:raise ValueError('Native history/control differs from frozen selection')
 from graphnav import stage1_runtime
 from graphnav.native_object_runtime import install_native_object_belief
 original=stage1_runtime.GraphNavStage1Policy
 class NativePolicy(original):
  def __init__(self,checkpoint,directory,**kwargs):
   super().__init__(checkpoint,directory,**kwargs)
   install_native_object_belief(self,a.native_checkpoint,checkpoint,control=a.native_control,history_window=a.native_history_window)
 stage1_runtime.GraphNavStage1Policy=NativePolicy
 output=value('--output')
 if output:
  output=Path(output)
  if '--resume-existing' in rest:
   if split!='val_seen' or not a.native_frozen_manifest:raise ValueError('Native resume requires frozen Seen candidate')
   from graphnav.navigation_resume import prepare_resume
   prepare_resume(output,a.native_frozen_manifest,checkpoint,value('--checkpoint'),a.native_history_window,a.native_control)
  elif output.exists():raise ValueError('Native navigation refuses existing output directory')
 sys.argv=[sys.argv[0]]+rest
 from scripts.evaluate_native_shared_closedloop import main as evaluate
 try:evaluate()
 finally:
  if output and output.exists():atomic_json(dict(kind='native-object-navigation-v1',native_checkpoint=str(checkpoint.resolve()),native_checkpoint_sha=file_sha256(checkpoint),control=a.native_control,history_window=a.native_history_window,split=split,diagnostic=split=='val_seen',metrics_written=(output/'metrics.json').exists()),output/'native_experiment.json')


if __name__=='__main__':main()
