import bin2py
import os
import numpy as np
import scipy.io as sio
from collections import namedtuple
import visionloader as vl
import electrode_map

PATH_MATLAB_LOADED_DATA_30UM_LTIKE = '/Volumes/Lab/Users/ericwu/infrastructure/30um_litke_testcases/2008-08-27-0-data000-cf-wueric-all_forpy_wueric.mat'
ANALYSIS_PATH_30UM_LITKE = '/Volumes/Lab/Users/ericwu/infrastructure/30um_litke_testcases/data000-cf-wueric-all-nodup'
ANALYSIS_DS_NAME_30UM_LITKE = 'data000-cf-wueric-all-nodup'

PATH_MATLAB_LOADED_DATA_HIERLEMANN = '/Volumes/Lab/Users/ericwu/infrastructure/simulated_hierlemann_testcases/2008-08-27-0-data000-cf-nodup_forpy_wueric.mat'
ANALYSIS_PATH_HIERLEMANN = '/Volumes/Lab/Users/ericwu/infrastructure/simulated_hierlemann_testcases/data000-cf'
ANALYSIS_DS_NAME_HIERLEMANN = 'data000-cf'

PATH_MATLAB_LOADED_DATA_60UM_LITKE = '/Volumes/Lab/Users/ericwu/infrastructure/60um_litke_testcases/2016-02-17-5-litkeformat.mat'
ANALYSIS_PATH_60UM_LITKE = '/Volumes/Lab/Users/ericwu/infrastructure/60um_litke_testcases/data000'
ANALYSIS_DS_NAME_60UM_LITKE = 'data000'

CellSTASpikeCount = namedtuple('CellSTASpikeCount', 
                               ['cell_id', 'center_x', 'center_y', 'sd_x', 'sd_y', 'rot_rad', 'nspikes'])



class SingleNeuronData:
    
    def __init__ (self, 
                  cell_id, 
                  center_x, 
                  center_y, 
                  sd_x, 
                  sd_y, 
                  rot_rad, 
                  nspikes, 
                  full_sta_movie, 
                  full_ei_movie,
                  timecourse):         
        self.cell_id = cell_id
        self.center_x = center_x
        self.center_y = center_y
        self.sd_x = sd_x
        self.sd_y = sd_y
        self.rot_rad = rot_rad
        self.nspikes = nspikes
        self.full_sta_movie = full_sta_movie
        self.full_ei_movie = full_ei_movie
        self.timecourse = timecourse
        
        
    def __hash__ (self):
        return hash(self.cell_id) # cell ids are guaranteed to be unique so we can hash them

DatasetSummary = namedtuple('DatasetSummary',
                           ['cell_by_cell_id', 'cell_types_dict'])

def parse_matlab_data (loaded_mat, include_ei=False):
    
    cell_types_dict = {}
    cell_by_cell_id = {}
    
    cell_id_to_index = {}
    
    cell_sta_fits = loaded_mat['temp'][()][0]
    cell_spikes = loaded_mat['temp'][()][1]
    cell_id_in_order = loaded_mat['temp'][()][2]
    cell_classes = loaded_mat['temp'][()][4]
    cell_sta_raw = loaded_mat['temp'][()][5]
    cell_timecourse_raw = loaded_mat['temp'][()][6]
    
    cell_ei = None
    if include_ei:
        cell_ei = loaded_mat['temp'][()][7][()][0]
    
    for i, cell_id in enumerate(cell_id_in_order):
        cell_id_to_index[cell_id] = i
    
    for i in range(cell_classes.shape[0]):
        
        cell_type = cell_classes[i][()][0]
        cell_type_ids = cell_classes[i][()][1]
        
        if type(cell_type) is str: # properly formed cell class name
        
            if cell_type not in cell_types_dict:

                cell_types_dict[cell_type] = []


            # grab the data for the cell            
            if type(cell_type_ids) is int:
                cell_type_ids = [cell_type_ids]
                
            for cell_id in cell_type_ids:
                sta_fit_data = cell_sta_fits[()][cell_id_to_index[cell_id]][()]
                sta_center = sta_fit_data[0]
                sta_sd = sta_fit_data[1]
                sta_angle = sta_fit_data[2]
                
                n_spikes = cell_spikes[()][cell_id_to_index[cell_id]].shape[0]
                
                sta_movie = cell_sta_raw[cell_id_to_index[cell_id]]
                
                timecourse_for_cell = cell_timecourse_raw[cell_id_to_index[cell_id]]
                
                ei_for_cell = None
                if include_ei:
                    ei_for_cell = cell_ei[cell_id_to_index[cell_id]]
                
                cell_data_packed = SingleNeuronData(cell_id,
                                                     sta_center[0],
                                                     sta_center[1],
                                                     sta_sd[0],
                                                     sta_sd[1],
                                                     sta_angle,
                                                     n_spikes,
                                                     sta_movie,
                                                     ei_for_cell,
                                                     timecourse_for_cell)
                
                cell_types_dict[cell_type].append(cell_data_packed)
                cell_by_cell_id[cell_id] = cell_data_packed

    return DatasetSummary(cell_by_cell_id, cell_types_dict)

def test_load_electrode_map_30um_litke ():

    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_30UM_LTIKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=False)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_30UM_LITKE,
                                ANALYSIS_DS_NAME_30UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)
    el_map_loaded = vcd_test.get_electrode_map()
    el_map_expected = electrode_map.LITKE_519_ARRAY_MAP

    for i in range(el_map_expected.shape[0]):
        assert el_map_loaded[i,0] == el_map_expected[i,0], "Electrode map doesn't match at index ({0},0)".format(i)
        assert el_map_loaded[i,1] == el_map_expected[i,1], "Electrode map doesn't match at index ({0},1)".format(i)

def test_load_cell_type_30um_litke ():
    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_30UM_LTIKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=False)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_30UM_LITKE,
                                ANALYSIS_DS_NAME_30UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=False)

    for cell_type, cell_list in dataset_parsed.cell_types_dict.items():
        for cell_data in cell_list:
            cell_id_expected = cell_data.cell_id

            cell_type_loaded_from_params = vcd_test.get_data_for_cell(cell_id_expected, 'classID')
            assert cell_type_loaded_from_params == cell_type, "Cell type {0} doesn't match expected {1}".format(cell_type_loaded_from_params, cell_type)


def test_load_sta_fit_30um_litke ():


    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_30UM_LTIKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=True)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_30UM_LITKE,
                                ANALYSIS_DS_NAME_30UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)

    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():

        sta_fit = vcd_test.get_stafit_for_cell(cell_id)
        center_x_loaded = sta_fit.center_x
        center_y_loaded = sta_fit.center_y
        sd_x_loaded = sta_fit.std_x
        sd_y_loaded = sta_fit.std_y
        rotation_loaded = sta_fit.rot

        assert center_x_loaded == cell_data.center_x, "X centers do not match"
        assert center_y_loaded == cell_data.center_y, "Y centers do not match"
        assert sd_x_loaded == cell_data.sd_x, "x stds do not match"
        assert sd_y_loaded == cell_data.sd_y, "y stds do not match"
        assert rotation_loaded == cell_data.rot_rad, "rotations do not match"


def test_load_ei_30um_litke ():


    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_30UM_LTIKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=True)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_30UM_LITKE,
                                ANALYSIS_DS_NAME_30UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)

    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():
        ei_for_cell_id = cell_data.full_ei_movie

        ei_to_test = vcd_test.get_data_for_cell(cell_id, 'EI').ei

        assert ei_to_test.shape == ei_for_cell_id.shape, "EI shape doesn't match up"

        for i in range(ei_for_cell_id.shape[0]):
            for j in range(ei_for_cell_id.shape[1]):
                assert ei_to_test[i,j] == ei_for_cell_id[i,j], "loaded data {0} does not match expected data {1}, index ({2},{3})".format(ei_to_test[i,j], ei_for_cell_id[i,j], i, j)


def test_load_timecourse_30um_litke ():
    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_30UM_LTIKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=True)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_30UM_LITKE,
                                ANALYSIS_DS_NAME_30UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)

    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():
        timecourses = cell_data.timecourse

        red_timecourse = timecourses[0]
        green_timecourse = timecourses[1]
        blue_timecourse = timecourses[2]

        red_timecourse_loaded = vcd_test.get_data_for_cell(cell_id, 'RedTimeCourse')
        green_timecourse_loaded = vcd_test.get_data_for_cell(cell_id, 'GreenTimeCourse')
        blue_timecourse_loaded = vcd_test.get_data_for_cell(cell_id, 'BlueTimeCourse')

        for i in range(red_timecourse.shape[0]):
            assert red_timecourse_loaded[i] == red_timecourse[i], "red timecourse doesn't match"
            assert green_timecourse_loaded[i] == green_timecourse[i], "green timecourse doesn't match"
            assert blue_timecourse_loaded[i] == blue_timecourse[i], "blue timecourse doesn't match"


def test_load_sta_30um_litke ():
    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_30UM_LTIKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=True)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_30UM_LITKE,
                                ANALYSIS_DS_NAME_30UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_sta=True)


    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():
        sta_raw = cell_data.full_sta_movie

        print(sta_raw.shape)
        red_sta = sta_raw[:,:,0,:]
        green_sta = sta_raw[:,:,1,:]
        blue_sta = sta_raw[:,:,2,:]

        sta_obj = vcd_test.get_data_for_cell(cell_id, 'STAraw')
        red_sta_loaded = sta_obj.red
        green_sta_loaded = sta_obj.green
        blue_sta_loaded = sta_obj.blue

        assert red_sta.shape == red_sta_loaded.shape, "Shapes don't match"

        for i in range(red_sta.shape[0]):
            for j in range(red_sta.shape[1]):
                for k in range(red_sta.shape[2]):
                    assert red_sta_loaded[i,j,k] == red_sta[i,j,k], "red sta doesn't match, index ({0},{1},{2})".format(i,j,k)
                    assert green_sta_loaded[i,j,k] == green_sta[i,j,k], "green sta doesn't match, index ({0},{1},{2})".format(i,j,k)
                    assert blue_sta_loaded[i,j,k] == blue_sta[i,j,k], "blue sta doesn't match, index ({0},{1},{2})".format(i,j,k)



def test_load_cell_type_hierlemann ():
    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_HIERLEMANN, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=False)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_HIERLEMANN,
                                ANALYSIS_DS_NAME_HIERLEMANN,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=False)

    for cell_type, cell_list in dataset_parsed.cell_types_dict.items():
        for cell_data in cell_list:
            cell_id_expected = cell_data.cell_id

            cell_type_loaded_from_params = vcd_test.get_data_for_cell(cell_id_expected, 'classID')
            assert cell_type_loaded_from_params == cell_type, "Cell type {0} doesn't match expected {1}".format(cell_type_loaded_from_params, cell_type)


def test_load_sta_fit_hierlemann ():


    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_HIERLEMANN, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=False)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_HIERLEMANN,
                                ANALYSIS_DS_NAME_HIERLEMANN,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)

    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():

        sta_fit = vcd_test.get_stafit_for_cell(cell_id)
        center_x_loaded = sta_fit.center_x
        center_y_loaded = sta_fit.center_y
        sd_x_loaded = sta_fit.std_x
        sd_y_loaded = sta_fit.std_y
        rotation_loaded = sta_fit.rot

        assert center_x_loaded == cell_data.center_x, "X centers do not match"
        assert center_y_loaded == cell_data.center_y, "Y centers do not match"
        assert sd_x_loaded == cell_data.sd_x, "x stds do not match"
        assert sd_y_loaded == cell_data.sd_y, "y stds do not match"
        assert rotation_loaded == cell_data.rot_rad, "rotations do not match"

def test_load_electrode_map_60um_litke ():

    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_60UM_LITKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=False)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_60UM_LITKE,
                                ANALYSIS_DS_NAME_60UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)

    el_map_loaded = vcd_test.get_electrode_map()
    el_map_expected = electrode_map.LITKE_512_ARRAY_MAP

    for i in range(el_map_expected.shape[0]):
        assert el_map_loaded[i,0] == el_map_expected[i,0], "Electrode map doesn't match at index ({0},0)".format(i)
        assert el_map_loaded[i,1] == el_map_expected[i,1], "Electrode map doesn't match at index ({0},1)".format(i)


def test_load_cell_type_60um_litke ():
    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_60UM_LITKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=False)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_60UM_LITKE,
                                ANALYSIS_DS_NAME_60UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=False)

    for cell_type, cell_list in dataset_parsed.cell_types_dict.items():
        for cell_data in cell_list:
            cell_id_expected = cell_data.cell_id

            cell_type_loaded_from_params = vcd_test.get_data_for_cell(cell_id_expected, 'classID')
            assert cell_type_loaded_from_params == cell_type, "Cell type {0} doesn't match expected {1}".format(cell_type_loaded_from_params, cell_type)


def test_load_sta_fit_60um_litke ():


    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_60UM_LITKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=True)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_60UM_LITKE,
                                ANALYSIS_DS_NAME_60UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)

    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():

        sta_fit = vcd_test.get_stafit_for_cell(cell_id)
        center_x_loaded = sta_fit.center_x
        center_y_loaded = sta_fit.center_y
        sd_x_loaded = sta_fit.std_x
        sd_y_loaded = sta_fit.std_y
        rotation_loaded = sta_fit.rot

        assert center_x_loaded == cell_data.center_x, "X centers do not match"
        assert center_y_loaded == cell_data.center_y, "Y centers do not match"
        assert sd_x_loaded == cell_data.sd_x, "x stds do not match"
        assert sd_y_loaded == cell_data.sd_y, "y stds do not match"
        assert rotation_loaded == cell_data.rot_rad, "rotations do not match"


def test_load_timecourse_60um_litke ():
    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_60UM_LITKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=True)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_60UM_LITKE,
                                ANALYSIS_DS_NAME_60UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)

    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():
        timecourses = cell_data.timecourse

        red_timecourse = timecourses[0]
        green_timecourse = timecourses[1]
        blue_timecourse = timecourses[2]

        red_timecourse_loaded = vcd_test.get_data_for_cell(cell_id, 'RedTimeCourse')
        green_timecourse_loaded = vcd_test.get_data_for_cell(cell_id, 'GreenTimeCourse')
        blue_timecourse_loaded = vcd_test.get_data_for_cell(cell_id, 'BlueTimeCourse')

        for i in range(red_timecourse.shape[0]):
            assert red_timecourse_loaded[i] == red_timecourse[i], "red timecourse doesn't match"
            assert green_timecourse_loaded[i] == green_timecourse[i], "green timecourse doesn't match"
            assert blue_timecourse_loaded[i] == blue_timecourse[i], "blue timecourse doesn't match"


def test_load_ei_60um_litke ():


    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_60UM_LITKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=True)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_60UM_LITKE,
                                ANALYSIS_DS_NAME_60UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_ei=True)

    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():
        ei_for_cell_id = cell_data.full_ei_movie

        ei_to_test = vcd_test.get_data_for_cell(cell_id, 'EI').ei

        assert ei_to_test.shape == ei_for_cell_id.shape, "EI shape doesn't match up"

        for i in range(ei_for_cell_id.shape[0]):
            for j in range(ei_for_cell_id.shape[1]):
                assert ei_to_test[i,j] == ei_for_cell_id[i,j], "loaded data {0} does not match expected data {1}, index ({2},{3}) for cell {4}".format(ei_to_test[i,j], ei_for_cell_id[i,j], i, j, cell_id)


def test_load_sta_60um_litke ():
    f_dataset = sio.loadmat(PATH_MATLAB_LOADED_DATA_60UM_LITKE, squeeze_me=True)
    dataset_parsed = parse_matlab_data(f_dataset, include_ei=True)

    vcd_test = vl.load_vision_data(ANALYSIS_PATH_60UM_LITKE,
                                ANALYSIS_DS_NAME_60UM_LITKE,
                                include_runtimemovie_params=True,
                                include_params=True,
                                include_sta=True)


    for cell_id, cell_data in dataset_parsed.cell_by_cell_id.items():
        sta_raw = cell_data.full_sta_movie

        print(sta_raw.shape)
        red_sta = sta_raw[:,:,0,:]
        green_sta = sta_raw[:,:,1,:]
        blue_sta = sta_raw[:,:,2,:]

        sta_obj = vcd_test.get_data_for_cell(cell_id, 'STAraw')
        red_sta_loaded = sta_obj.red
        green_sta_loaded = sta_obj.green
        blue_sta_loaded = sta_obj.blue

        assert red_sta.shape == red_sta_loaded.shape, "Shapes don't match"


        for i in range(red_sta.shape[0]):
            for j in range(red_sta.shape[1]):
                for k in range(red_sta.shape[2]):
                    assert red_sta_loaded[i,j,k] == red_sta[i,j,k], "red sta doesn't match, index ({0},{1},{2})".format(i,j,k)
                    assert green_sta_loaded[i,j,k] == green_sta[i,j,k], "green sta doesn't match, index ({0},{1},{2})".format(i,j,k)
                    assert blue_sta_loaded[i,j,k] == blue_sta[i,j,k], "blue sta doesn't match, index ({0},{1},{2})".format(i,j,k)
