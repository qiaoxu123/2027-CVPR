"""Frozen, local-only BERT semantic features. Reads public inference inputs only."""
import argparse,collections,os,time
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from common import *

def encoder_text(inp):
 text=inp['instruction']
 for h in sorted(inp['mentioned'],key=lambda x:x['start'],reverse=True):
  cats={inp['scene'][i]['category'] for i in h['resolved_ids'] if i in inp['scene']};category=next(iter(cats)).lower() if len(cats)==1 else 'landmark'
  text=text[:h['start']]+' named '+category+' '+text[h['end']:]
 # Names and IDs are not features; relation role and scoped text are.
 scope=inp['instruction'];a,b=inp['direction_span'];left=max(scope.rfind('.',0,a),scope.rfind(';',0,a))+1;right=scope.find('.',b);right=len(scope) if right<0 else right
 scoped=scope[left:right]
 # The scope string is masked independently to avoid revealing original names.
 for h in sorted(inp['mentioned'],key=lambda x:x['start'],reverse=True):
  if left<=h['start']<h['end']<=right:
   lo,hi=h['start']-left,h['end']-left;scoped=scoped[:lo]+' named landmark '+scoped[hi:]
 return 'relation '+inp['word']+' . '+scoped+' [SEP] '+text

def run(out,device):
 import torch,transformers
 from transformers import AutoModel,AutoTokenizer
 torch.set_num_threads(4);X=readlines(out/'public_inputs.jsonl.gz');sentences=[encoder_text(x) for x in X];unique=list(dict.fromkeys(sentences));snap=Path('/home/20261024_1/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
 tokenizer=AutoTokenizer.from_pretrained(snap,local_files_only=True);model=AutoModel.from_pretrained(snap,local_files_only=True).to(device).eval();emb=[];start=time.time()
 with torch.inference_mode():
  for j in range(0,len(unique),32):
   batch=tokenizer(unique[j:j+32],padding=True,truncation=True,max_length=192,return_tensors='pt');batch={k:v.to(device) for k,v in batch.items()};z=model(**batch).last_hidden_state;m=batch['attention_mask'].unsqueeze(-1);vec=(z*m).sum(1)/m.sum(1);emb.append(vec.cpu().numpy().astype('float32'))
 A=np.concatenate(emb);lookup={s:i for i,s in enumerate(unique)};B=A[[lookup[s] for s in sentences]];np.savez_compressed(out/'bert_features.npz',features=B,ids=np.array([x['occurrence_id'] for x in X]))
 writej(out/'bert_feature_manifest.json',{'model':'bert-base-uncased','snapshot':snap.name,'weights_sha256':sha(snap/'model.safetensors'),'config_sha256':sha(snap/'config.json'),'local_files_only':True,'frozen':True,'fine_tuned':False,'device':device,'torch_version':torch.__version__,'transformers_version':transformers.__version__,'max_tokens':192,'pooling':'attention-mask mean of final hidden states','entity_names_masked':True,'inputs':'relation scope + full instruction ONLY; no coordinates, GT, identities, labels or trajectories','public_input_sha256':sha(out/'public_inputs.jsonl.gz'),'n_relations':len(X),'n_unique_masked_texts':len(unique),'features_shape':list(B.shape),'elapsed_seconds':time.time()-start})
 writelines(out/'encoder_texts.jsonl.gz',[{'occurrence_id':x['occurrence_id'],'encoder_text':s} for x,s in zip(X,sentences)]);print(readj(out/'bert_feature_manifest.json'),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();run(BASE/'results',a.device)
