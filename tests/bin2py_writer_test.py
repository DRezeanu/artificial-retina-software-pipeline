import bin2py
import os
import numpy as np


def test_write_and_readback_519 (tmpdir):

    header_dummy_519 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 1',
                                                                        'test-case-1',
                                                                        1,
                                                                        1504,
                                                                        520,
                                                                        20000,
                                                                        100)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_519, os.path.join(tmpdir.dirname, 'testcase1'),'data000') as pbfw:
        pbfw.write_samples(correct_output_from_matlab)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase1'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase1', 'data000'))
    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase1', 'data000', 'data000000.bin'))



    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase1', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 100) as pbfr:
        assert len(pbfr.bin_filestream_list) == 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 1 '
        assert header.dataset_identifier == 'test-case-1'
        assert header.format == 1
        assert header.array_id == 1504
        assert header.num_electrodes == 520
        assert header.frequency == 20000
        assert header.n_samples == 100

        read_back = pbfr.get_data(0, 100)
        for i in range(100):
            for j in range(520):
                assert read_back[i,j] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)

def test_write_and_readback_519_row_major (tmpdir):

    header_dummy_519 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 1',
                                                                        'test-case-1',
                                                                        1,
                                                                        1504,
                                                                        520,
                                                                        20000,
                                                                        100)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_519, os.path.join(tmpdir.dirname, 'testcase101'),
                                'data000',
                                is_row_major=True) as pbfw:
        pbfw.write_samples(correct_output_from_matlab.T)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase101'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase101', 'data000'))
    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase1', 'data000', 'data000000.bin'))

    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase101', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 100, is_row_major=True) as pbfr:
        assert len(pbfr.bin_filestream_list) == 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 1 '
        assert header.dataset_identifier == 'test-case-1'
        assert header.format == 1
        assert header.array_id == 1504
        assert header.num_electrodes == 520
        assert header.frequency == 20000
        assert header.n_samples == 100

        read_back = pbfr.get_data(0, 100)

        assert read_back.shape == (520, 100)

        for i in range(100):
            for j in range(520):
                assert read_back[j,i] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)


def test_write_and_readback_519_write_chunk_prime (tmpdir):

    header_dummy_519 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 2',
                                                                        'test-case-2',
                                                                        1,
                                                                        1504,
                                                                        520,
                                                                        20000,
                                                                        100)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_519, os.path.join(tmpdir.dirname, 'testcase2'),
                                'data000',
                                bin_file_n_samples=17) as pbfw:
        pbfw.write_samples(correct_output_from_matlab)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase2'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase2', 'data000'))
    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase2', 'data000', 'data000000.bin'))


    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase2', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 100) as pbfr:
        assert len(pbfr.bin_filestream_list) == 100 // 17 + 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 2 '
        assert header.dataset_identifier == 'test-case-2'
        assert header.format == 1
        assert header.array_id == 1504
        assert header.num_electrodes == 520
        assert header.frequency == 20000
        assert header.n_samples == 100

        read_back = pbfr.get_data(0, 100)
        for i in range(100):
            for j in range(520):
                assert read_back[i,j] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)

def test_write_and_readback_519_write_chunk_prime_row_major (tmpdir):

    header_dummy_519 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 2',
                                                                        'test-case-2',
                                                                        1,
                                                                        1504,
                                                                        520,
                                                                        20000,
                                                                        100)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_519, os.path.join(tmpdir.dirname, 'testcase202'),
                                'data000',
                                bin_file_n_samples=17,
                                is_row_major=True) as pbfw:
        pbfw.write_samples(correct_output_from_matlab.T)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase202'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase202', 'data000'))
    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase202', 'data000', 'data000000.bin'))


    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase202', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 100, is_row_major=True) as pbfr:
        assert len(pbfr.bin_filestream_list) == 100 // 17 + 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 2 '
        assert header.dataset_identifier == 'test-case-2'
        assert header.format == 1
        assert header.array_id == 1504
        assert header.num_electrodes == 520
        assert header.frequency == 20000
        assert header.n_samples == 100

        read_back = pbfr.get_data(0, 100)
        for i in range(100):
            for j in range(520):
                assert read_back[j,i] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)


def test_write_and_readback_519_read_write_chunk_prime (tmpdir):

    header_dummy_519 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 3',
                                                                        'test-case-3',
                                                                        1,
                                                                        1504,
                                                                        520,
                                                                        20000,
                                                                        100)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_519, os.path.join(tmpdir.dirname, 'testcase3'),'data000', bin_file_n_samples=17) as pbfw:
        pbfw.write_samples(correct_output_from_matlab)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase3'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase3', 'data000'))
    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase3', 'data000', 'data000000.bin'))



    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase3', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 19) as pbfr:
        assert len(pbfr.bin_filestream_list) == 100 // 17 + 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 3 '
        assert header.dataset_identifier == 'test-case-3'
        assert header.format == 1
        assert header.array_id == 1504
        assert header.num_electrodes == 520
        assert header.frequency == 20000
        assert header.n_samples == 100

        read_back = pbfr.get_data(0, 100)
        for i in range(100):
            for j in range(520):
                assert read_back[i,j] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)

def test_write_and_readback_519_read_write_chunk_prime_row_major (tmpdir):

    header_dummy_519 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 3',
                                                                        'test-case-3',
                                                                        1,
                                                                        1504,
                                                                        520,
                                                                        20000,
                                                                        100)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_519, os.path.join(tmpdir.dirname, 'testcase303'),
                                'data000',
                                bin_file_n_samples=17,
                                is_row_major=True) as pbfw:
        pbfw.write_samples(correct_output_from_matlab.T)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase303'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase303', 'data000'))
    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase303', 'data000', 'data000000.bin'))



    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase303', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 19, is_row_major=True) as pbfr:
        assert len(pbfr.bin_filestream_list) == 100 // 17 + 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 3 '
        assert header.dataset_identifier == 'test-case-3'
        assert header.format == 1
        assert header.array_id == 1504
        assert header.num_electrodes == 520
        assert header.frequency == 20000
        assert header.n_samples == 100

        read_back = pbfr.get_data(0, 100)
        for i in range(100):
            for j in range(520):
                assert read_back[j,i] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)


def test_write_and_readback_512 (tmpdir):
    header_dummy_512 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 11',
                                                                        'test-case-11',
                                                                        1,
                                                                        504,
                                                                        513,
                                                                        20000,
                                                                        500)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_512, os.path.join(tmpdir.dirname, 'testcase11'),'data000') as pbfw:
        pbfw.write_samples(correct_output_from_matlab)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase11'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase11', 'data000'))

    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase11', 'data000', 'data000000.bin'))


    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase11', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 500) as pbfr:
        assert len(pbfr.bin_filestream_list) == 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 11'
        assert header.dataset_identifier == 'test-case-11'
        assert header.format == 1
        assert header.array_id == 504
        assert header.num_electrodes == 513
        assert header.frequency == 20000
        assert header.n_samples == 500

        read_back = pbfr.get_data(0, 500)
        for i in range(500):
            for j in range(513):
                assert read_back[i,j] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)

def test_write_and_readback_512_row_major (tmpdir):
    header_dummy_512 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 11',
                                                                        'test-case-11',
                                                                        1,
                                                                        504,
                                                                        513,
                                                                        20000,
                                                                        500)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_512, os.path.join(tmpdir.dirname, 'testcase11011'),
                                'data000',
                                is_row_major=True) as pbfw:
        pbfw.write_samples(correct_output_from_matlab.T)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase11011'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase11011', 'data000'))

    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase11011', 'data000', 'data000000.bin'))


    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase11011', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 500, is_row_major=True) as pbfr:
        assert len(pbfr.bin_filestream_list) == 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 11'
        assert header.dataset_identifier == 'test-case-11'
        assert header.format == 1
        assert header.array_id == 504
        assert header.num_electrodes == 513
        assert header.frequency == 20000
        assert header.n_samples == 500

        read_back = pbfr.get_data(0, 500)
        for i in range(500):
            for j in range(513):
                assert read_back[j,i] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)


def test_write_and_readback_512_write_prime (tmpdir):
    header_dummy_512 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 12',
                                                                        'test-case-12',
                                                                        1,
                                                                        504,
                                                                        513,
                                                                        20000,
                                                                        500)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_512, os.path.join(tmpdir.dirname, 'testcase12'),'data000', bin_file_n_samples=17) as pbfw:
        pbfw.write_samples(correct_output_from_matlab)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase12'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase12', 'data000'))

    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase12', 'data000', 'data000000.bin'))


    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase12', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 500) as pbfr:
        assert len(pbfr.bin_filestream_list) == 500 // 17 + 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 12'
        assert header.dataset_identifier == 'test-case-12'
        assert header.format == 1
        assert header.array_id == 504
        assert header.num_electrodes == 513
        assert header.frequency == 20000
        assert header.n_samples == 500

        read_back = pbfr.get_data(0, 500)
        for i in range(500):
            for j in range(513):
                assert read_back[i,j] == correct_output_from_matlab[i,j], \
                    'data is incorrect, indices ({0},{1})'.format(i,j)

def test_write_and_readback_512_write_prime_row_major (tmpdir):
    header_dummy_512 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 12',
                                                                        'test-case-12',
                                                                        1,
                                                                        504,
                                                                        513,
                                                                        20000,
                                                                        500)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_512, os.path.join(tmpdir.dirname, 'testcase12012'),
                                'data000',
                                bin_file_n_samples=17,
                                is_row_major=True) as pbfw:
        pbfw.write_samples(correct_output_from_matlab.T)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase12012'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase12012', 'data000'))

    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase12012', 'data000', 'data000000.bin'))


    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase12012', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 500, is_row_major=True) as pbfr:
        assert len(pbfr.bin_filestream_list) == 500 // 17 + 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 12'
        assert header.dataset_identifier == 'test-case-12'
        assert header.format == 1
        assert header.array_id == 504
        assert header.num_electrodes == 513
        assert header.frequency == 20000
        assert header.n_samples == 500

        read_back = pbfr.get_data(0, 500)
        for i in range(500):
            for j in range(513):
                assert read_back[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)


def test_write_and_readback_512_read_write_prime (tmpdir):
    header_dummy_512 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 13',
                                                                        'test-case-13',
                                                                        1,
                                                                        504,
                                                                        513,
                                                                        20000,
                                                                        500)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_512, os.path.join(tmpdir.dirname, 'testcase13'),'data000', bin_file_n_samples=17) as pbfw:
        pbfw.write_samples(correct_output_from_matlab)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase13'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase13', 'data000'))


    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase13', 'data000', 'data000000.bin'))

    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase13', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 19) as pbfr:
        assert len(pbfr.bin_filestream_list) == 500 // 17 + 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 13'
        assert header.dataset_identifier == 'test-case-13'
        assert header.format == 1
        assert header.array_id == 504
        assert header.num_electrodes == 513
        assert header.frequency == 20000
        assert header.n_samples == 500

        read_back = pbfr.get_data(0, 500)
        for i in range(500):
            for j in range(513):
                assert read_back[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)


def test_write_and_readback_512_read_write_prime_row_major (tmpdir):
    header_dummy_512 = bin2py.PyBinHeader.make_header_from_parameters(1904,
                                                                        0,
                                                                        'test case 13',
                                                                        'test-case-13',
                                                                        1,
                                                                        504,
                                                                        513,
                                                                        20000,
                                                                        500)


    correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')

    with bin2py.PyBinFileWriter(header_dummy_512, os.path.join(tmpdir.dirname, 'testcase13013'),
                                'data000',
                                bin_file_n_samples=17,
                                is_row_major=True) as pbfw:
        pbfw.write_samples(correct_output_from_matlab.T)


    # make sure the appropriate directories were created
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase13013'))
    assert os.path.isdir(os.path.join(tmpdir.dirname, 'testcase13013', 'data000'))


    assert os.path.exists(os.path.join(tmpdir.dirname, 'testcase13013', 'data000', 'data000000.bin'))

    # now attempt reading back the data
    path_to_readback = os.path.join(tmpdir.dirname, 'testcase13013', 'data000')
    with bin2py.PyBinFileReader(path_to_readback, chunk_samples = 19, is_row_major=True) as pbfr:
        assert len(pbfr.bin_filestream_list) == 500 // 17 + 1


        # make sure the header is right
        header = pbfr.header
        assert header.time_base == 1904
        assert header.seconds_time == 0
        assert header.comment == 'test case 13'
        assert header.dataset_identifier == 'test-case-13'
        assert header.format == 1
        assert header.array_id == 504
        assert header.num_electrodes == 513
        assert header.frequency == 20000
        assert header.n_samples == 500

        read_back = pbfr.get_data(0, 500)
        for i in range(500):
            for j in range(513):
                assert read_back[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)
