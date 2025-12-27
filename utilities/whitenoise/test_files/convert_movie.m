% add path

addpath('/Volumes/Lab/Users/ericwu/matlab/code/lab');

% load the vision data (load a random dataset, we don't care which
% we just want synthetic trigger times
datarun = load_data('/Volumes/Analysis/2017-11-29-0/data001/data001');
datarun = load_neurons(datarun);
datarun = load_params(datarun);

% get the movie given the vision data
[mov, height, width, duration, refresh] = get_movie('/Volumes/Analysis/stimuli/white-noise-xml/RGB-16-2-0.48-11111.xml', datarun.triggers, 200);
% mov has shape (height, width, 3, duration)
h5create('RGB-16-2-0.48-11111.hdf5', '/movie', size(mov));
h5write('RGB-16-2-0.48-11111.hdf5', '/movie', mov);




% get the movie given the vision data
[mov, height, width, duration, refresh] = get_movie('/Volumes/Analysis/stimuli/white-noise-xml/RGB-8-2-0.48-11111.xml', datarun.triggers, 200);
% mov has shape (height, width, 3, duration)
h5create('RGB-8-2-0.48-11111.hdf5', '/movie', size(mov));
h5write('RGB-8-2-0.48-11111.hdf5', '/movie', mov);



% get the movie given the vision data
[mov, height, width, duration, refresh] = get_movie('/Volumes/Analysis/stimuli/white-noise-xml/RGB-4-2-0.48-11111.xml', datarun.triggers, 200);
% mov has shape (height, width, 3, duration)
h5create('RGB-4-2-0.48-11111.hdf5', '/movie', size(mov));
h5write('RGB-4-2-0.48-11111.hdf5', '/movie', mov);




% get the movie given the vision data
[mov, height, width, duration, refresh] = get_movie('/Volumes/Analysis/stimuli/white-noise-xml/BW-8-2-0.48-11111.xml', datarun.triggers, 200);
% mov has shape (height, width, 3, duration)
h5create('BW-8-2-0.48-11111.hdf5', '/movie', size(mov));
h5write('BW-8-2-0.48-11111.hdf5', '/movie', mov);



% get the movie given the vision data
[mov, height, width, duration, refresh] = get_movie('/Volumes/Analysis/stimuli/white-noise-xml/RGB-4-4-0.48-22222.xml', datarun.triggers, 200);
% mov has shape (height, width, 3, duration)
h5create('RGB-4-4-0.48-22222.hdf5', '/movie', size(mov));
h5write('RGB-4-4-0.48-22222.hdf5', '/movie', mov);




% get the movie given the vision data
[mov, height, width, duration, refresh] = get_movie('/Volumes/Analysis/stimuli/white-noise-xml/RGB-8-2-0.48-11111-40x40.xml', datarun.triggers, 200);
% mov has shape (height, width, 3, duration)
h5create('RGB-8-2-0.48-11111-40x40.hdf5', '/movie', size(mov));
h5write('RGB-8-2-0.48-11111-40x40.hdf5', '/movie', mov);
