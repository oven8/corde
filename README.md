# corde
[`corde`](https://github.com/oven8/corde) is a fast CORSIKA de-thinner program for python that can be used to regenerate longitudinal data for lost samples.

## Install
Currently [`corde`](https://github.com/oven8/corde) can be installed via
```
pip install git+https://github.com/oven8/corde
```

## Features
[`corde`](https://github.com/oven8/corde) can load multiple files at once
```
import corde as cd

file_dir = '/home/user/Corsika_Files/'
file_list = ['DAT000001','DAT000002','DAT000003','DAT000004','DAT000005','DAT000006','DAT000007','DAT000008','DAT000009','DAT000010']
dethinner = cd.corsika_dethin(file_dir,file_list)
```
Regeneration happens automatically on initialization!
> [!WARNING]
> Note that [`corde`](https://github.com/oven8/corde) can only read thin type binary files in CORSIKA.

[`corde`](https://github.com/oven8/corde) also supports storing the regenerated data in HDF5 format
```
import corde as cd

file_dir = '/home/user/Corsika_Files/'
file_list = ['DAT000001','DAT000002','DAT000003','DAT000004','DAT000005','DAT000006','DAT000007','DAT000008','DAT000009','DAT000010']
dethinner = cd.corsika_dethin(file_dir,file_list,storage='HDF5')

```
The file structure for a single event is given below
```
Group: event1
  Dataset: azimuth | Shape: () | Type: float32
  Dataset: del_t | Shape: (3406,) | Type: float32
  Dataset: dmax_list | Shape: (3406,) | Type: float32
  Dataset: pi | Shape: (3406, 3) | Type: float32
  Dataset: pid | Shape: (3406,) | Type: int64
  Dataset: regen_pid | Shape: (315759,) | Type: int64
  Dataset: regen_xi | Shape: (315759, 3) | Type: float64
  Dataset: thinning_weight | Shape: (3406,) | Type: float32
  Dataset: xi | Shape: (3406, 3) | Type: float32
  Dataset: z | Shape: () | Type: float32
  Dataset: zenith | Shape: () | Type: float32
```
`corde.util` and `corde.histogram` contains many useful functions that can be used in analysis and multiparameter binning.
