"""Experimental public OSM vector preprocessing; never called by V2/V3 runtime."""
import hashlib,json
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape,mapping
from shapely.ops import transform

def convert_public_osm(geojson,registration):
    """Refuse unvalidated registrations; convert WGS84 vectors into local XY."""
    if not registration.get('verified') or registration.get('validation_n',0)<3:
        raise ValueError('Independent geographic validation required')
    errors=np.asarray(registration.get('validation_errors_m',[]),float)
    if len(errors)<3 or not np.isfinite(errors).all() or errors.max()>5:
        raise ValueError('Geographic validation exceeds fixed 5m visualization limit')
    mat=np.asarray(registration['matrix'],float)
    if mat.shape!=(2,3) or not np.isfinite(mat).all() or abs(np.linalg.det(mat[:,:2]))<1e-8:raise ValueError('Invalid transform')
    inverse=np.linalg.inv(mat[:,:2]);offset=mat[:,2]
    projector=Transformer.from_crs('EPSG:4326',registration['projected_crs'],always_xy=True)
    def to_xy(x,y,z=None):
        e,n=projector.transform(x,y);xy=inverse@(np.array([e,n])-offset[:,None] if np.ndim(e) else np.array([e,n])-offset)
        return xy[0],xy[1]
    out=[]
    for f in geojson.get('features',[]):
        tags=f.get('properties',{});kind=('parking' if tags.get('amenity')=='parking' else 'road' if tags.get('highway') else 'building' if tags.get('building') else 'place' if tags.get('amenity') or tags.get('landuse') else 'other')
        g=transform(to_xy,shape(f['geometry']))
        if g.is_empty or not g.is_valid:continue
        out.append(dict(id=f.get('id',tags.get('@id')),kind=kind,name=tags.get('name',''),geometry=mapping(g),tags=tags,area_m2=g.area,length_m=g.length))
    edges=[]
    for i,a in enumerate(out):
        ga=shape(a['geometry'])
        for b in out[i+1:]:
            gb=shape(b['geometry']);distance=ga.distance(gb)
            if ga.contains(gb):edges.append(dict(subject=a['id'],relation='contains',object=b['id']))
            elif gb.contains(ga):edges.append(dict(subject=b['id'],relation='contains',object=a['id']))
            elif ga.intersects(gb):edges.append(dict(subject=a['id'],relation='intersects',object=b['id']))
            if distance<=10:edges.append(dict(subject=a['id'],relation='near_10m',object=b['id'],distance_m=distance))
    return dict(features=out,relations=edges,scope='public vector geometry proxies, not learned language reasoning',registration_sha256=hashlib.sha256(json.dumps(registration,sort_keys=True).encode()).hexdigest())
