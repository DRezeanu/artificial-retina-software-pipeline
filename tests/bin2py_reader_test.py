import bin2py
import numpy as np


########################
# test PyBinHeader

class TestPyBinHeader ():

    def test_one_519_array (self):
        # 519 array tests
        first_bin0_path = '/Volumes/Data/2016-02-17-4/data000/data000000.bin'

        with open(first_bin0_path, 'rb') as fbp:

            pbh = bin2py.PyBinHeader.construct_from_binfile(fbp)
            assert pbh.time_base == 1904
            assert pbh.seconds_time == 3538619866
            assert pbh.dataset_identifier == '2017-02-17-4-data000'
            assert pbh.n_samples == 36000000
            assert pbh.num_electrodes == 520
            assert pbh.frequency == 20000
            assert pbh.format == 1
            assert pbh.header_length == 168
            assert pbh.comment == 'RGB-25-1-0.48-11111 oled'
            assert pbh.array_id == 1504

    def test_two_519_array (self):

        second_bin0_path = '/Volumes/Data/2018-11-12-2/data000/data000000.bin'
        with open(second_bin0_path, 'rb') as sbp:
            pbh2 = bin2py.PyBinHeader.construct_from_binfile(sbp)

            assert pbh2.time_base == 1904
            assert pbh2.seconds_time == 3624942441
            assert pbh2.dataset_identifier == '2018-11-12-2-data000'
            assert pbh2.n_samples == 36000000
            assert pbh2.num_electrodes == 520
            assert pbh2.frequency == 20000
            assert pbh2.format == 1
            assert pbh2.header_length == 180
            assert pbh2.comment == 'RGB-20-1-0.48-11111-40x30 oled NDF0 '
            assert pbh2.array_id == 1504


    def test_one_512_array (self):
        third_bin0_path = '/Volumes/Data/2016-02-17-8/data000/data000000.bin'
        with open(third_bin0_path, 'rb') as tbp:

            pbh3 = bin2py.PyBinHeader.construct_from_binfile(tbp)

            assert pbh3.time_base == 1904
            assert pbh3.seconds_time == 3538758754
            assert pbh3.dataset_identifier == '2016-02-17-8-data000'
            assert pbh3.n_samples == 36000000
            assert pbh3.num_electrodes == 513
            assert pbh3.frequency == 20000
            assert pbh3.format == 1
            assert pbh3.header_length == 162
            assert pbh3.comment == 'RGB-8-2-0.48-11111'
            assert pbh3.array_id == 504

    def test_two_512_array (self):

        fourth_bin0_path = '/Volumes/Data/2018-03-01-0/data010/data010000.bin'
        with open(fourth_bin0_path, 'rb') as fbp:

            pbh4 = bin2py.PyBinHeader.construct_from_binfile(fbp)

            assert pbh4.time_base == 1904
            assert pbh4.seconds_time == 3602813716
            assert pbh4.dataset_identifier == '2018-03-01-0-data010'
            assert pbh4.n_samples == 18000000
            assert pbh4.num_electrodes == 513
            assert pbh4.frequency == 20000
            assert pbh4.format == 1
            assert pbh4.header_length == 164
            assert pbh4.comment == 'RGB-16-2-0.48-11111 '
            assert pbh4.array_id == 504


class TestPyBinFileReader:

    def test_read_519_first100_samples_one_electrode (self):

        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 780
            assert pbfr.decoder._N_ELECTRODES == 520

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            # test an odd channel
            output = pbfr.get_data_for_electrode (15, 0, 100)
            assert output.shape == (100, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(100):
                assert output[i] == correct_output_from_matlab[i,15], 'data is incorrect'

            # test an even channel
            output = pbfr.get_data_for_electrode (248, 0, 100)
            assert output.shape == (100, ), 'shape is incorrect'
            for i in range(100):
                assert output[i] == correct_output_from_matlab[i,248], 'data is incorrect'


            # test the TTL channel
            output = pbfr.get_data_for_electrode (0, 0, 100)
            assert output.shape == (100, ), 'shape is incorrect'
            for i in range(100):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect'

    def test_read_519_first100_samples_one_electrode_row_major(self):

        path1 = '/Volumes/Data/2016-02-17-4/data000'

        with bin2py.PyBinFileReader(path1, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 780
            assert pbfr.decoder._N_ELECTRODES == 520

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            # test an odd channel
            output = pbfr.get_data_for_electrode (15, 0, 100)
            assert output.shape == (100, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(100):
                assert output[i] == correct_output_from_matlab[i,15], 'data is incorrect'

            # test an even channel
            output = pbfr.get_data_for_electrode (248, 0, 100)
            assert output.shape == (100, ), 'shape is incorrect'
            for i in range(100):
                assert output[i] == correct_output_from_matlab[i,248], 'data is incorrect'


            # test the TTL channel
            output = pbfr.get_data_for_electrode (0, 0, 100)
            assert output.shape == (100, ), 'shape is incorrect'
            for i in range(100):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect'

    def test_read_519_first_500_samples_one_electrode_wackochunk (self):

        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 780
            assert pbfr.decoder._N_ELECTRODES == 520

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            # test an odd channel
            output = pbfr.get_data_for_electrode (15, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,15], 'data is incorrect'

            # test an even channel
            output = pbfr.get_data_for_electrode (248, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,248], 'data is incorrect'


            # test the TTL channel
            output = pbfr.get_data_for_electrode (0, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect'


    def test_read_519_first_500_samples_one_electrode_wackochunk_row_major (self):

        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 780
            assert pbfr.decoder._N_ELECTRODES == 520

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            # test an odd channel
            output = pbfr.get_data_for_electrode (15, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,15], 'data is incorrect'

            # test an even channel
            output = pbfr.get_data_for_electrode (248, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,248], 'data is incorrect'


            # test the TTL channel
            output = pbfr.get_data_for_electrode (0, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect'

    def test_read_519_filejump_500_samples_one_electrode (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=50) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

    
            # test odd channel
            output = pbfr.get_data_for_electrode(15, 2400000 - 200, 500)
            assert output.shape == (500, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-midfile500.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,15], 'data is incorrect, indices ({0},{1})'.format(i,15)

            # test even channel
            output = pbfr.get_data_for_electrode(510, 2400000 - 200, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,510], 'data is incorrect, indices ({0},{1})'.format(i,510)

            # test TTL channel
            output = pbfr.get_data_for_electrode(0, 2400000 - 200, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect, indices ({0},{1})'.format(i,0)

    def test_read_519_filejump_500_samples_one_electrode_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=50, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

            # test odd channel
            output = pbfr.get_data_for_electrode(15, 2400000 - 200, 500)
            assert output.shape == (500,), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-midfile500.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i, 15], 'data is incorrect, indices ({0},{1})'.format(i,
                                                                                                                     15)

            # test even channel
            output = pbfr.get_data_for_electrode(510, 2400000 - 200, 500)
            assert output.shape == (500,), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i, 510], 'data is incorrect, indices ({0},{1})'.format(i,
                                                                                                                      510)

            # test TTL channel
            output = pbfr.get_data_for_electrode(0, 2400000 - 200, 500)
            assert output.shape == (500,), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i, 0], 'data is incorrect, indices ({0},{1})'.format(i,
                                                                                                                    0)

    def test_read_519_first100_samples (self):

        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 780
            assert pbfr.decoder._N_ELECTRODES == 520

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 100)
            assert output.shape == (100, 520), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-100samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(100):
                for j in range(520):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect'

    def test_read_519_first_500_samples_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=100, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16


            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 780
            assert pbfr.decoder._N_ELECTRODES == 520

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 500)
            assert output.shape == (520, 500), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(520):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)


    def test_read_519_first_500_samples (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=100) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16


            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 780
            assert pbfr.decoder._N_ELECTRODES == 520

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 500)
            assert output.shape == (500, 520), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(520):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

    def test_read_519_first_500_samples_wackochunk (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 500)
            assert output.shape == (500, 520), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(520):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

    def test_read_519_first_500_samples_wackochunk_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 500)
            assert output.shape == (520, 500), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(520):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

    def test_read_519_filejump_500_samples_wackochunk (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(2400000 - 200, 500)
            assert output.shape == (500, 520), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-midfile500.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(520):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

            output = pbfr.get_data(2400000 - 200, 500)
            assert output.shape == (500, 520), 'shape is incorrect on second read attempt'
            for i in range(500):
                for j in range(520):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect on second read attempt, indices ({0},{1})'.format(i,j)

    def test_read_519_filejump_500_samples_wackochunk_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(2400000 - 200, 500)
            assert output.shape == (520, 500), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-midfile500.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(520):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

            output = pbfr.get_data(2400000 - 200, 500)
            assert output.shape == (520, 500), 'shape is incorrect on second read attempt'
            for i in range(500):
                for j in range(520):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect on second read attempt, indices ({0},{1})'.format(i,j)

    def test_read_519_filejump_500_samples (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=50) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(2400000 - 200, 500)
            assert output.shape == (500, 520), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-midfile500.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(520):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

    def test_read_519_filejump_500_samples_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-4/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=50, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

            output = pbfr.get_data(2400000 - 200, 500)
            assert output.shape == (520, 500), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-4-data000-midfile500.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(520):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)


    def test_read_512_first_500_samples_one_electrode (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=100) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16


            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513


            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

            # test odd channel
            output = pbfr.get_data_for_electrode(3, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,3], 'data is incorrect, indices ({0},{1})'.format(i,3)

            # test even channel
            output = pbfr.get_data_for_electrode(48, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,48], 'data is incorrect, indices ({0},{1})'.format(i,48)

            # test TTL channel
            output = pbfr.get_data_for_electrode(0, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect, indices ({0},{1})'.format(i,0)

    def test_read_512_first_500_samples_one_electrode_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=100, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16


            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513


            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

            # test odd channel
            output = pbfr.get_data_for_electrode(3, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,3], 'data is incorrect, indices ({0},{1})'.format(i,3)

            # test even channel
            output = pbfr.get_data_for_electrode(48, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,48], 'data is incorrect, indices ({0},{1})'.format(i,48)

            # test TTL channel
            output = pbfr.get_data_for_electrode(0, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect, indices ({0},{1})'.format(i,0)

    def test_read_512_first_500_samples_one_electrode_wackochunk (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16


            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513


            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

            # test odd channel
            output = pbfr.get_data_for_electrode(3, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,3], 'data is incorrect, indices ({0},{1})'.format(i,3)

            # test even channel
            output = pbfr.get_data_for_electrode(48, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,48], 'data is incorrect, indices ({0},{1})'.format(i,48)

            # test TTL channel
            output = pbfr.get_data_for_electrode(0, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect, indices ({0},{1})'.format(i,0)


    def test_read_512_first_500_samples_one_electrode_wackochunk_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16


            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513


            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

            # test odd channel
            output = pbfr.get_data_for_electrode(3, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,3], 'data is incorrect, indices ({0},{1})'.format(i,3)

            # test even channel
            output = pbfr.get_data_for_electrode(48, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,48], 'data is incorrect, indices ({0},{1})'.format(i,48)

            # test TTL channel
            output = pbfr.get_data_for_electrode(0, 0, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect, indices ({0},{1})'.format(i,0)


    def test_read_512_filejump_500_samples_one_electrode (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=50) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-filejump.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')

            output = pbfr.get_data_for_electrode(3, 2400000 - 100, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,3], 'data is incorrect, indices ({0},{1})'.format(i,3)

            # test even channel
            output = pbfr.get_data_for_electrode(48, 2400000 - 100, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,48], 'data is incorrect, indices ({0},{1})'.format(i,48)

            # test TTL channel
            output = pbfr.get_data_for_electrode(0, 2400000 - 100, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect, indices ({0},{1})'.format(i,0)

    def test_read_512_filejump_500_samples_one_electrode_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=50, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-filejump.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')

            output = pbfr.get_data_for_electrode(3, 2400000 - 100, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,3], 'data is incorrect, indices ({0},{1})'.format(i,3)

            # test even channel
            output = pbfr.get_data_for_electrode(48, 2400000 - 100, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,48], 'data is incorrect, indices ({0},{1})'.format(i,48)

            # test TTL channel
            output = pbfr.get_data_for_electrode(0, 2400000 - 100, 500)
            assert output.shape == (500, ), 'shape is incorrect'
            for i in range(500):
                assert output[i] == correct_output_from_matlab[i,0], 'data is incorrect, indices ({0},{1})'.format(i,0)


    def test_read_512_first_500_samples (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=100) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16


            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513


            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 500)
            assert output.shape == (500, 513), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(513):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

    def test_read_512_first_500_samples_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=100, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16


            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513


            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 500)
            assert output.shape == (513, 500), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(513):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

    def test_read_512_first_500_samples_wackochunk (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 500)
            assert output.shape == (500, 513), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(513):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)


            output = pbfr.get_data(0, 500)
            assert output.shape == (500, 513), 'shape is incorrect on second read attempt'
            for i in range(500):
                for j in range(513):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect on second read attempt, indices ({0},{1})'.format(i,j)


    def test_read_512_filejump_500_samples_wackochunk (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(2400000 - 100, 500)
            assert output.shape == (500, 513), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-filejump.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(513):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

            output = pbfr.get_data(2400000 - 100, 500)
            assert output.shape == (500, 513), 'shape is incorrect on second read attempt'

            for i in range(500):
                for j in range(513):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect on second read attempt, indices ({0},{1})'.format(i,j)

    def test_read_512_first_500_samples_wackochunk_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=17, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16
            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(0, 500)
            assert output.shape == (513,500), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-500samples.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(513):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)


            output = pbfr.get_data(0, 500)
            assert output.shape == (513,500), 'shape is incorrect on second read attempt'
            for i in range(500):
                for j in range(513):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect on second read attempt, indices ({0},{1})'.format(i,j)

    def test_read_512_filejump_500_samples (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=50) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(2400000 - 100, 500)
            assert output.shape == (500, 513), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-filejump.csv', 
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(513):
                    assert output[i,j] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

    def test_read_512_filejump_500_samples_row_major (self):
        path1 = '/Volumes/Data/2016-02-17-8/data000'
        with bin2py.PyBinFileReader(path1, chunk_samples=50, is_row_major=True) as pbfr:
            assert len(pbfr.bin_filestream_list) == 15
            assert len(pbfr.sample_number_list) == 16

            assert pbfr.decoder._N_BYTES_PER_SAMPLE == 770
            assert pbfr.decoder._N_ELECTRODES == 513

            assert pbfr.sample_number_list[0] == 0
            assert pbfr.sample_number_list[-1] == pbfr.header.n_samples


            output = pbfr.get_data(2400000 - 100, 500)
            assert output.shape == (513,500), 'shape is incorrect'

            correct_output_from_matlab = np.loadtxt('bin2py_test_data/2016-02-17-8-data000-filejump.csv',
                                                    dtype=np.int16,
                                                    delimiter=',')
            for i in range(500):
                for j in range(513):
                    assert output[j,i] == correct_output_from_matlab[i,j], 'data is incorrect, indices ({0},{1})'.format(i,j)

