<!--
 * @Author: HenryVarro666 1504517223@qq.com
 * @Date: 1969-12-31 19:00:00
 * @LastEditors: HenryVarro666 1504517223@qq.com
 * @LastEditTime: 2024-06-25 20:33:06
 * @FilePath: /DensityMap+GNN/gyrianalyzer/import.md
-->


>import vtk
import nibabel.freesurfer.io as io
from collections import defaultdict
import pdb
import numpy as np
import os
import shutil
import time
import networkx as nx
import argparse






---
```shell
python3 /home/lab/Documents/HCP_data_32k_flip/gyralnet.py \
--root_dir="$PWD" \
--subject_list_start_id="$subject_id" \
--subject_list_end_id="$((subject_id + 1))" \
--input_dir="${subject_id}_recon" \
--out_dir='gyralnet_island_32k_flip'
```