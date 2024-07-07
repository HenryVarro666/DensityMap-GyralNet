#!/bin/bash
###
 # @Author: HenryVarro666 1504517223@qq.com
 # @Date: 1969-12-31 19:00:00
 # @LastEditors: HenryVarro666 1504517223@qq.com
 # @LastEditTime: 2024-07-05 06:47:03
 # @FilePath: /DensityMap-GyralNet/sbtach.sh
### 

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=50g
#SBATCH -n 40
#SBATCH -c 32
#SBATCH -t 2-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chaocaog@ad.unc.edu

module add python
python3 gyralnet.py