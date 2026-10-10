from scripts.prepare_roads import clip_segment, convert_roads


def test_clipped_real_road_segment_remains_aligned():
    bounds=[0,0,400,400]
    assert clip_segment([-20,100],[100,100],bounds)==[[0.0,100.0],[100.0,100.0]]
    assert clip_segment([-20,500],[100,500],bounds) is None


def test_geojson_converter_filters_outside_and_preserves_road_names():
    features=[
        {"type":"Feature","id":13,"properties":{"map_name":"city_block_0",
         "name":"Main Road","osm_id":"way-1"},
         "geometry":{"type":"LineString","coordinates":[[-20,100],[200,100],[420,100]]}},
        {"type":"Feature","id":14,"properties":{"map_name":"city_block_0",
         "name":"Outside"},"geometry":{"type":"LineString",
         "coordinates":[[500,500],[600,600]]}},
    ]
    roads=convert_roads(features,{"city_block_0":[0,0,400,400]})
    assert len(roads["city_block_0"])==2
    assert all(x["name"]=="Main Road" for x in roads["city_block_0"])
    assert all(x["coordinate_system"]=="world" for x in roads["city_block_0"])
