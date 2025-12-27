import numpy as np

import h5py

#from . import random_noise as rn
import whitenoise.random_noise as rn

class TestRandomNoise:

    def test_rgb_8_2_048_11111 (self):

        N_FRAMES = 50

        rn_obj = rn.RandomNoiseFrameGenerator.construct_from_xml('/Volumes/Analysis/stimuli/white-noise-xml/RGB-8-2-0.48-11111.xml')
        first_50_frames = rn_obj.generate_block_of_frames(N_FRAMES)

        assert first_50_frames.shape == (50, 40, 80, 3)

        with h5py.File('/Volumes/Lab/Users/ericwu/artificial-retina-software-pipeline/utilities/whitenoise/test_files/RGB-8-2-0.48-11111.hdf5', 'r') as f:
            xx = np.array(f['/movie'])
            f.close()

        xx_int = (xx * 256).astype(np.uint8)
        matched_shape = np.transpose(xx_int, (0, 3, 2, 1))

        for i in range(first_50_frames.shape[0]):
            for j in range(first_50_frames.shape[1]):
                for k in range(first_50_frames.shape[2]):
                    for l in range(first_50_frames.shape[3]):
                        assert matched_shape[i,j,k,l] == first_50_frames[i,j,k,l]

    def test_rgb_4_2_048_11111 (self):

        N_FRAMES = 50

        rn_obj = rn.RandomNoiseFrameGenerator.construct_from_xml('/Volumes/Analysis/stimuli/white-noise-xml/RGB-4-2-0.48-11111.xml')
        first_50_frames = rn_obj.generate_block_of_frames(N_FRAMES)

        with h5py.File('/Volumes/Lab/Users/ericwu/artificial-retina-software-pipeline/utilities/whitenoise/test_files/RGB-4-2-0.48-11111.hdf5', 'r') as f:
            xx = np.array(f['/movie'])
            f.close()

        xx_int = (xx * 256).astype(np.uint8)
        matched_shape = np.transpose(xx_int, (0, 3, 2, 1))

        for i in range(first_50_frames.shape[0]):
            for j in range(first_50_frames.shape[1]):
                for k in range(first_50_frames.shape[2]):
                    for l in range(first_50_frames.shape[3]):
                        assert matched_shape[i,j,k,l] == first_50_frames[i,j,k,l]


    def test_bw_8_2_048_11111 (self):

        N_FRAMES = 50

        rn_obj = rn.RandomNoiseFrameGenerator.construct_from_xml('/Volumes/Analysis/stimuli/white-noise-xml/BW-8-2-0.48-11111.xml')
        first_50_frames = rn_obj.generate_block_of_frames(N_FRAMES)

        with h5py.File('/Volumes/Lab/Users/ericwu/artificial-retina-software-pipeline/utilities/whitenoise/test_files/BW-8-2-0.48-11111.hdf5', 'r') as f:
            xx = np.array(f['/movie'])
            f.close()

        xx_int = (xx * 256).astype(np.uint8)
        matched_shape = np.transpose(xx_int, (0, 3, 2, 1))

        for i in range(first_50_frames.shape[0]):
            for j in range(first_50_frames.shape[1]):
                for k in range(first_50_frames.shape[2]):
                    for l in range(first_50_frames.shape[3]):
                        assert matched_shape[i,j,k,l] == first_50_frames[i,j,k,l]


    def test_rgb_16_2_048_11111 (self):

        N_FRAMES = 50

        rn_obj = rn.RandomNoiseFrameGenerator.construct_from_xml('/Volumes/Analysis/stimuli/white-noise-xml/RGB-16-2-0.48-11111.xml')
        first_50_frames = rn_obj.generate_block_of_frames(N_FRAMES)

        with h5py.File('/Volumes/Lab/Users/ericwu/artificial-retina-software-pipeline/utilities/whitenoise/test_files/RGB-16-2-0.48-11111.hdf5', 'r') as f:
            xx = np.array(f['/movie'])
            f.close()

        xx_int = (xx * 256).astype(np.uint8)
        matched_shape = np.transpose(xx_int, (0, 3, 2, 1))

        for i in range(first_50_frames.shape[0]):
            for j in range(first_50_frames.shape[1]):
                for k in range(first_50_frames.shape[2]):
                    for l in range(first_50_frames.shape[3]):
                        assert matched_shape[i,j,k,l] == first_50_frames[i,j,k,l]


    def test_rgb_8_2_048_11111_40_40 (self):

        N_FRAMES = 50

        rn_obj = rn.RandomNoiseFrameGenerator.construct_from_xml('/Volumes/Analysis/stimuli/white-noise-xml/RGB-8-2-0.48-11111-40x40.xml')
        first_50_frames = rn_obj.generate_block_of_frames(N_FRAMES)

        with h5py.File('/Volumes/Lab/Users/ericwu/artificial-retina-software-pipeline/utilities/whitenoise/test_files/RGB-8-2-0.48-11111-40x40.hdf5', 'r') as f:
            xx = np.array(f['/movie'])
            f.close()

        xx_int = (xx * 256).astype(np.uint8)
        matched_shape = np.transpose(xx_int, (0, 3, 2, 1))

        for i in range(first_50_frames.shape[0]):
            for j in range(first_50_frames.shape[1]):
                for k in range(first_50_frames.shape[2]):
                    for l in range(first_50_frames.shape[3]):
                        assert matched_shape[i,j,k,l] == first_50_frames[i,j,k,l]


    def test_rgb_4_4_048_22222 (self):

        N_FRAMES = 50

        rn_obj = rn.RandomNoiseFrameGenerator.construct_from_xml('/Volumes/Analysis/stimuli/white-noise-xml/RGB-4-4-0.48-22222.xml')
        first_50_frames = rn_obj.generate_block_of_frames(N_FRAMES)

        with h5py.File('/Volumes/Lab/Users/ericwu/artificial-retina-software-pipeline/utilities/whitenoise/test_files/RGB-4-4-0.48-22222.hdf5', 'r') as f:
            xx = np.array(f['/movie'])
            f.close()

        xx_int = (xx * 256).astype(np.uint8)
        matched_shape = np.transpose(xx_int, (0, 3, 2, 1))

        for i in range(first_50_frames.shape[0]):
            for j in range(first_50_frames.shape[1]):
                for k in range(first_50_frames.shape[2]):
                    for l in range(first_50_frames.shape[3]):
                        assert matched_shape[i,j,k,l] == first_50_frames[i,j,k,l]


    def test_advance_frames_rgb_4_4_048_11111(self):
        N_FRAMES = 50

        rn_obj = rn.RandomNoiseFrameGenerator.construct_from_xml(
            '/Volumes/Analysis/stimuli/white-noise-xml/RGB-8-2-0.48-11111-40x40.xml')
        first_50_frames = rn_obj.generate_block_of_frames(N_FRAMES)

        rn_obj.reset_seed_to_beginning()
        rn_obj.advance_seed_n_frames(49)

        frame_to_test = rn_obj.generate_next_frame()
        for i in range(frame_to_test.shape[0]):
            for j in range(frame_to_test.shape[1]):
                for k in range(frame_to_test.shape[2]):
                    assert frame_to_test[i,j,k] == first_50_frames[49,i,j,k]
