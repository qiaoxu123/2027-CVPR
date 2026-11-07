"""Public-name balanced allocation under a fixed Polygon token budget."""
from collections import OrderedDict

def balanced_landmark_slots(landmarks,capacity=16):
 if capacity<1:raise ValueError('positive capacity required')
 if len(landmarks)<=capacity:return list(landmarks)
 groups=OrderedDict()
 for item in landmarks:groups.setdefault(item['name'],[]).append(item)
 selected=[];depth=0
 while len(selected)<capacity:
  added=False
  for group in groups.values():
   if depth<len(group):selected.append(group[depth]);added=True
   if len(selected)==capacity:break
  if not added:break
  depth+=1
 return selected
