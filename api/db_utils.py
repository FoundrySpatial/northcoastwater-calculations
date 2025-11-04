# Array index corresponds to zoom level.
# See https://docs.mapbox.com/help/glossary/zoom-level/#zoom-levels-and-geographical-distance for definition.
meters_per_pixel = [
    59959.436 ,
    29979.718 ,
    14989.859 ,
    7494.929,
    3747.465,
    1873.732 ,
    936.866 ,
    468.433 ,
    234.217,
    117.108 ,
    58.554 ,
    29.277,
    14.639,
    7.319,
    3.660,
    1.830,
    0.915,
    0.457,
    0.229
]

def get_search_distance(zoom_level):
    tile_length_pixels = 512
    if (zoom_level >= 19):
        # Using 100 as a minimum search distance, regardless of zoom
        return 100
    if (zoom_level >= 0):
        return tile_length_pixels * meters_per_pixel[zoom_level]
    return None

# Test syncing behavior!