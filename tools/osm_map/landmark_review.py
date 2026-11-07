"""Review correspondence identities without promoting them to geographic validation."""
import fcntl
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
from fastapi import HTTPException, Request
from rasterio.warp import transform


def install(app, store):
    store = Path(store)
    root = store / 'landmark_matches'
    reviews = store / 'landmark_reviews.json'

    def records():
        p = root / 'matches.json'
        return json.loads(p.read_text()) if p.exists() else []

    def decisions():
        return json.loads(reviews.read_text()) if reviews.exists() else {}

    def status(r, d):
        return d.get(r['match_id'], {}).get('decision', 'pending' if r['high_confidence'] else 'rejected_automatic')

    @app.get('/api/landmark-matches')
    def listing(city: str = '', state: str = ''):
        rows = records(); d = decisions()
        filtered = [r for r in rows if (not city or r['city'] == city) and (not state or status(r, d) == state)]
        keys = ['match_id', 'city', 'map_name', 'kind', 'confidence', 'split', 'independence_group', 'representative', 'anchor_residual_m', 'source_name', 'osm_name', 'reasons']
        return dict(total=len(rows), counts={s: sum(status(r, d) == s for r in rows) for s in ['pending', 'confirmed', 'rejected', 'rejected_automatic']}, rows=[dict({k: r.get(k) for k in keys}, review_status=status(r, d)) for r in filtered], warning='自动残差仅为代理；确认对应身份不会自动批准地理配准，不改变拟合/验证分组。')

    @app.get('/api/landmark-matches/{match_id}')
    def detail(match_id: str):
        r = next((r for r in records() if r['match_id'] == match_id), None)
        if r is None: raise HTTPException(404)
        summary = json.loads((root / 'summary.json').read_text())
        m = np.array(summary['cities'][r['city']]['matrix'])
        def convert(coords):
            a = np.array(coords, dtype=float)
            x, y = transform('EPSG:27700', 'EPSG:4326', a[:, 0], a[:, 1])
            return list(zip(y, x))
        polygon = r.get('osm_polygon_en')
        rings = polygon.get('coordinates', []) if polygon and polygon['type'] == 'Polygon' else []
        osm_ll = [convert(ring) for ring in rings]
        osm_local = [((np.array(ring) - m[:, 2]) @ np.linalg.inv(m[:, :2]).T).tolist() for ring in rings]
        return dict(r, osm_latlon_rings=osm_ll, osm_local_rings=osm_local, osm_anchor_latlon=convert([r['osm_anchor_en']])[0], city_matrix=m.tolist(), review=decisions().get(match_id), review_status=status(r, decisions()))

    @app.post('/api/landmark-matches/{match_id}/review')
    async def review(match_id: str, request: Request):
        r = next((r for r in records() if r['match_id'] == match_id), None)
        if r is None: raise HTTPException(404)
        payload = await request.json()
        if payload.get('decision') not in ['confirmed', 'rejected', 'pending']: raise HTTPException(400, 'Invalid decision')
        note = str(payload.get('note', ''))
        if len(note) > 2000: raise HTTPException(400, 'Note too long')
        with (store / 'landmark_reviews.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            d = decisions(); old = d.get(match_id)
            d[match_id] = dict(decision=payload['decision'], note=note, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), previous=old, split=r['split'], independence_group=r['independence_group'])
            fd, tmp = tempfile.mkstemp(dir=store, prefix='.review-')
            try:
                with os.fdopen(fd, 'w') as f: json.dump(d, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
                os.replace(tmp, reviews)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        return dict(saved=True, review=d[match_id], geographic_validation_approved=False)
