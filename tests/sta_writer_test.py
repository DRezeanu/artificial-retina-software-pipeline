import os
import numpy as np

import visionloader as vl
import visionwriter as vw


def test_writeback_sta(tmpdir):

    with vl.STAReader('/Volumes/Lab/Users/ericwu/yass-reconstruction//2018-08-07-5/data000',
                      'data000') as star:

        stas_by_cell_id_read = star.get_all_stas_by_cell_id()
        width = star.width
        height = star.height
        depth = star.depth
        refresh_time = star.refresh_time
        sta_offset = star.sta_offset
        stixel_size = star.stixel_size

    tmp_path_to_write = os.path.join(tmpdir, 'data000')
    os.makedirs(tmp_path_to_write, exist_ok=True)

    with vw.STAWriter(tmp_path_to_write,
                      'data000',
                      width,
                      height,
                      depth,
                      refresh_time,
                      sta_offset,
                      stixel_size) as staw:
        staw.write_sta_by_cell_id(stas_by_cell_id_read)

    # now read back and check whether things are the same
    with vl.STAReader(tmp_path_to_write, 'data000') as star2:
        stas_by_cell_id_readback = star2.get_all_stas_by_cell_id()
        rb_width = star2.width
        rb_height = star2.height
        rb_depth = star2.depth
        rb_refresh_time = star2.refresh_time
        rb_sta_offset = star2.sta_offset
        rb_stixel_size = star2.stixel_size

    assert rb_width == width
    assert rb_height == height
    assert rb_depth == depth
    assert rb_refresh_time == refresh_time
    assert rb_sta_offset == sta_offset
    assert rb_stixel_size == stixel_size

    for cell_id, orig_stacont in stas_by_cell_id_read.items():
        assert cell_id in stas_by_cell_id_readback

        readback_stacont = stas_by_cell_id_readback[cell_id]

        # verify that the values check out
        assert np.allclose(orig_stacont.red, readback_stacont.red)
        assert np.allclose(orig_stacont.red_error, readback_stacont.red_error)

        assert np.allclose(orig_stacont.green, readback_stacont.green)
        assert np.allclose(orig_stacont.green_error, readback_stacont.green_error)

        assert np.allclose(orig_stacont.blue, readback_stacont.blue)
        assert np.allclose(orig_stacont.blue_error, readback_stacont.blue_error)

    return True
