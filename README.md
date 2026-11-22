# The EPIMETHEUS code (v0.1)

## Brief description
EPIMETHEUS simulates the propagation of epidemics on a (customizable) network and includes various sources of heterogeneities in the host population. This repository contains the code and the analysis notebook and scripts used for the paper "Turbulent Epidemics: Scale-dependent Fluctuations as a Probe of Population Heterogeneity" (Mueller, Lorke, Gomes & Padberg).

## Setup
The code requires a current Python and Jupyter installation. It was run using Python 3.12 on macOS Tahoe 26.6, and was also tested on Windows and Linux platforms. Some standard Python packages are required.

The code will attempt to download the requisite case data and geodata automatically. The analysis notebook will attempt to read saved pickled dumps of simulations, and will run them if these files are not present.

NOTE: Random number generators are initialized deterministically for the runs in "Turbulent Epidemics", but platform independence has not been verified.

# Contact
[Bernhard Mueller](bernhard.mueller@monash.edu)

