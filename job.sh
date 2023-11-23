#!/bin/sh
###job name
#BSUB -J painn_128_5

###Specify output-file
#BSUB -o painn_%J.out

###Specify error-file
#BSUB -e painn_%J.err

###specify queue
#BSUB -q gpuv100

###specify gpu
#BSUB -gpu "num=1:mode=exclusive_process"

### specify number of cores
#BSUB -n 8

###specify the amount of memory pr. core
#BSUB -R "rusage[mem=5G]"

###Specify that it only spans default one host
#BSUB -R "span[hosts=1]"

###Specify walltime
#BSUB -W 24:00

###Specify notifications:
#BSUB -B
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.9.1-python-3.9.14

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source painn-env/bin/activate

python_script="main.py"
python "$python_script"
