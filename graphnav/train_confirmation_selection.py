"""Validate a frozen TRAIN physical-target manifest without inspecting positions."""


def selected_train_tasks(raw_records, catalog, manifest):
    result=[];seen=set()
    for selected in manifest:
        index=selected['index']
        if not isinstance(index,int) or not 0<=index<len(raw_records):
            raise ValueError('invalid frozen TRAIN index')
        raw=raw_records[index]
        name=f"{raw['area']}_block_{raw['block']}"
        oid=int(raw['object_ids'][0]);group=(name,oid)
        expected=(selected['physical_group'][0],int(selected['physical_group'][1]))
        category=catalog[name][str(oid)]['object_type']
        if group!=expected or selected.get('category',category)!=category:
            raise ValueError('frozen TRAIN identity/category mismatch')
        if group in seen:raise ValueError('duplicate physical target')
        seen.add(group);result.append(((category,name,oid),index))
    if not result:raise ValueError('empty TRAIN selection')
    return sorted(result,key=lambda x:(x[0][1],x[1]))
