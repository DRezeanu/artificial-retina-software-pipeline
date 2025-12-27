# Quick instructions for Python .bin file reading and writing, Vision file reading and writing, stimulus frame reading

## Installation 
There are two ways to install this module, either (1) as a Python package in the
associated with a particular Python interpreter / conda environment, or (2) by compiling 
dependencies and adding this directory to your PYTHONPATH.

No matter which option you choose, you must run the setup script. This package has Cython and C++ dependencies, and the 
dependencies need to be compiled. 

It is highly recommended that you pick the same option for every environment that you intend to use, to avoid confusion
about which version of the code is being executed.

#### Dependencies
* numpy
* scipy
* statsmodels
* pybind11
* Cython

#### Installing the module as a package

In your conda environment, make sure that you have Cython and pybind11 installed. After doing so,
run

```shell script
python setup.py install
```

This installs the module like any other package you download from the internet, and no modifications
to your PYTHONPATH are necessary. This installation is associated with the specific conda interpreter
that you ran the installation with. If you need this to run in another conda environment, you must also
install it in the same way.

#### Compiling the dependencies and adding to PYTHONPATH

The alternative approach is to just compile the dependencies, and point the PYTHONPATH to include this folder

Compile the dependencies using the below

```shell script
python setup.py build_ext --inplace
```

Then add this folder (artificial-retina-software-pipeline/utilities) to your PYTHONPATH.

## Reading existing .bin files


To load blocks of (including TTL channel) data from all electrodes,

```python
import bin2py

with bin2py.PyBinFileReader('/Volumes/Data/9999-99-99-9/data000', chunk_samples=10000) as pbfr:
    samples = pbfr.get_data(start_sample_num, num_samples)
    
```

`chunk_samples` is the blocksize of samples that is loaded in a single
disk access. Its value does not affect the result of `get_data()`, but may
affect the speed (if too large or too small performance will be slow).
10000 works well for reading every channel on 512/519 data.

`samples` has shape `(num_samples, num_channels)` where `num_channels`
includes the TTL channel. The TTL channel is channel 0.

To load data for a single channel,


```python
import bin2py


with bin2py.PyBinFileReader('/Volumes/Data/9999-99-99-9/data000', chunk_samples=10000) as pbfr: 
    samples_one_channel = pbfr.get_data_for_electrode(channel_num, start_sample_num, num_samples)
    
```
`samples_one_channel` has shape `(num_samples, )`, and the TTL channel is
`channel_num = 0`.

Note that .bin files are electrode configuration agnostic, and the
formatting only depends on whether or not there are an even or odd number
of channels.

## Writing .bin files

In order to write .bin files, an appropriate .bin file header must first
be generated. This header stores the array id, number of channels, sample
rate, etc.

Then, given the header, .bin files containing sample data can be written.

Example, writing a fake 519 array .bin file:

```python
import bin2py

header = bin2py.PyBinHeader.make_519_header(0, 0, '', '', 0, n_samples_total)

with bin2py.PyBinFileWriter(header, 
                            '/Volumes/Scratch/9999-99-99-9/data000', 
                            'data000', 
                            bin_file_n_samples=2400000) as pbfw:
    pbfw.write_samples(data_array) 
    pbfw.write_samples(more_data_array)

``` 
`data_array` must have shape `(samples, num_channels)` where
`num_channels` includes the TTL channel. You can write repeatedly to the
same .bin file/folder.

This code automatically partitions the written data into multiple distinct
files in the same way that the recorded data is partitioned (the first
file contains the header, the remaining do not). You can change the number
of samples per file by setting optional parameter `bin_file_n_samples`. By default it is
2400000 samples per file.


## Reading Vision files

To load everything for a given analysis into Python, do the following:

```python
import visionloader as vl

analysis_data = vl.load_vision_data('/Volumes/Analysis/9999-99-99-9',
                                    'data000',
                                    include_params=True,
                                    include_ei=True,
                                    include_sta=True,
                                    include_neurons=True)
```

`analysis_data` is an object of type `vl.VisionCellDataTable`, and
functions as a table where you can grab data given a cell id.

### Reading from `vl.VisionCellDataTable` objects

You can get a complete list of cell ids by doing the following:

```python
cell_id_list = analysis_data.get_cell_ids()
```

You can get all cell ids of a given cell type by doing

```python
on_parasol_ids = analysis.get_all_cells_of_type('ON parasol')
```

where the parameter is the cell type name set in the Vision GUI.

We can read the following fields for each cell id from Vision from the `vl.VisionCellDataTable object`:
1. Spike times (from .neurons) 
2. EI and EI error (from .ei)
3. Contamination (from .params)
4. ACF (from .params)
5. Cell type classifications (from .params)
6. Full STA movie (from .sta)
7. STA fits (from .params)

There are dedicated methods for reading this data (i.e. `get_sta_for_cell()`, `get_ei_for_cell()`, etc.) See the source code comments for detailed documentation.

You can manually load .txt files with cell type classifications by doing the following:
```python
analysis_data.update_cell_type_classifications_from_text_file('/Volumes/Analysis/9999-99-99-9/data0000/wueric-cf.txt')
```

This updates the `vl.VisionCellDataTable` to contain the cell type
classifications from the specified text file rather than the .params file.

### Loading EIs 

Using the `vl.VisionCellDataTable` object, EIs can be loaded with
```python
import visionloader as vl

analysis_data = vl.load_vision_data('/Volumes/Analysis/9999-99-99-9',
                                    'data000',
                                    include_ei=True)
ei_container_for_cell_id = analysis_data.get_ei_for_cell(cell_id)
```

`ei_container_for_cell_id` is an `EIContainer` namedtuple with the followings fields:
* `ei` : the full EI, as a 2D matrix with shape `(n_electrodes, n_samples)`
* `ei_error`: EI error as a 2D matrix, as calculated by Vision
*  `nl_points`: number of samples before the spike, set in Vision
* `nr_points`: number of samples after the spike, set in Vision
* `n_samples_total`: number of samples in the EI, equal to `nl_points + nr_points + 1`

Alternatively, EIs for every cell can be loaded directly from the .ei file as follows:

```python
import visionloader as vl
with vl.EIReader('/Volumes/Analysis/2019-01-01-1/data000', 'data000') as eir:
    eis_by_cell_id = eir.get_all_eis_by_cell_id()
```

`eis_by_cell_id` is a Python dict, where the keys are integer cell ids and the values are `EIContainer` namedtuples.

### Loading full STAs
or to load STA information,

```python
import visionloader as vl
analysis_data = vl.load_vision_data('/Volumes/Analysis/9999-99-99-9',
                                    'data000',
                                    include_sta=True)
sta_container_for_cell_id = analysis_data.get_sta_for_cell(cell_id)
```

`sta_container_for_cell_id` is a `STAContainer` namedtuple with the following fields:

* `stixel_size`, the stixel size
* `refresh_time`, the monitor refresh time
* `red`, the red component of the STA as a`np.ndarray`, with shape `(width, height, num_frames)`
* `red_error`, the error in the red component of the STA
* `green`, the green component of the STA as a `np.ndarray`, with shape `(width, height, num_frames)`
* `green_error`, the error in the green component of the STA
* `blue`, the blue component of the STA as a `np.ndarray`, with shape `(width, height, num_frames)`
* `blue_error`, the error in the blue component of the STA

You can also directly load the STAs from the .sta file by doing the following:

```python
import visionloader as vl

with STAReader('/Volumes/Analysis/2019-01-01-1/data000', 'data000') as star:
    stas_by_cell_id = star.get_all_stas_by_cell_id()
```

`stas_by_cell_id` is a Python dict, where the keys are integer cell ids, and the values are `STAContainer` namedtuples.



## Writing Vision files

We can write .neurons and .globals files directly from Python. This
enables us to use Vision to calculate STAs and EIs as well as to classify
cell types even if we use some arbitrary non-Vision spikesorter.

In order to use Vision to do these things, you must generate both
a .neurons file (telling us what cell spiked at what time, as well as when
the TTL pulses were) as well as a .globals file.

```python
import visionwriter as vw

with vw.NeuronsFileWriter('/Volumes/Scratch/9999-99-99-9/data000', 'data000') as nfw:
    nfw.write_neuron_file(spike_times_by_cell_id, ttl_times, n_samples)

with vw.GlobalsFileWriter('/Volumes/Scratch/9999-99-99-9/data000', 'data000') as gfw:
    gfw.write_simplified_litke_array_globals_file(array_id, 0, 0, '', '', 0, n_samples)
```

`spike_times_by_cell_id` is a dict mapping integer cell ids to 1D
`np.ndarrays` containing the spike times of the corresponding cell.
`ttl_times` is a 1D `np.ndarray` containing the TTL trigger times from the
recording, and `n_samples` is the total number of samples in the
recording.

## Reading .rawMovie files

*.rawMovie files can be read as follows

```python
from rawmovie import RawMovieReader

with RawMovieReader('path/to/raw/movie') as rmr:
    all_frames = rmr.get_all_frames()

    some_frames = rmr.get_frame_sequence(start_frame_num, end_frame_num)
```

## Generating frames of white noise from XML

Note: This has only been tested with RGB stimulus. BW should work...

Frames of white noise can be generated from the Vision XML files as follows:

```python
from whitenoise import RandomNoiseFrameGenerator

frame_generator = RandomNoiseFrameGenerator.construct_from_xml('path/to/stimulus/xml')

single_frame = frame_generator.generate_next_frame() # shape (height, width, 3)

block_of_frames = frame_generator.generate_block_of_frames(n_frames) # shape (width, height, width, 3)
```

Note that ```frame_generator``` keeps the random seed as internal state, and each call to ```generate_next_frame``` or 
```generate_block_of_frames``` advances that state forward, such that subsequent calls get the next frames. Consecutive calls
of ```generate_next_frame``` and ```generate_block_of_frames``` return different arrays, because they correspond to different
stimulus frames.

```frame_generator``` can be reset to the beginning with ```frame_generator.reset_seed_to_beginning()```.

You can use the constructor for ```RandomNoiseFrameGenerator```; see the source code for the arguments.

### Software TODO list

* Handle bin file reading when the array ID is wrong
