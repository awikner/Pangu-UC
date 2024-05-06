import sys
import subprocess
import shlex
sys.path.append('/glade/work/awikner/Pangu-UC')
var_dict = {'data_dir': '/Users/Alexander/Documents/PLASIM/data/test_data',
            'boundary_dir': 'boundary_vars',
            'upper_air_variables': ['ta', 'ua', 'va', 'hus', 'clw'],
            'surface_variables': ['pl', 'tas'],
            'constant_boundary_variables': ['lsm', 'z0', 'sg'],
            'varying_boundary_variables': ['sst', 'rsdt', 'sic'],
            'train_year_start': 100,
            'train_year_end': 104,
            'val_year_start': 104,
            'val_year_end': 105,
            'surface_mean': 'plasim_test_51_150_surface_mean.nc',
            'surface_std': 'plasim_test_51_150_surface_std.nc',
            'upper_air_mean': 'plasim_test_51_150_mean.nc',
            'upper_air_std': 'plasim_test_51_150_std.nc',
            'boundary_mean': 'plasim_test_51_150_boundary_mean.nc',
            'boundary_std': 'plasim_test_51_150_boundary_std.nc',
            'calendar': 'proleptic_gregorian',
            'timedelta_hours': 6,
            'batch_size': 4,
}
arglist = []
for key, var in zip(var_dict.keys(), var_dict.values()):
    if type(var) == list:
        arglist.append('--%s %s' % (key, ' '.join([elem for elem in var])))
    else:
        arglist.append(f'--{key}={var}')
argstr = ' '.join(arglist)
print(argstr)
subprocess.run(shlex.split('python ../../../examples/pangu_plasim/train_nc.py %s' % argstr))

