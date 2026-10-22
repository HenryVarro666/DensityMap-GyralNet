<div align="center">

# DensityMap-GyralNet

**English** · [简体中文](README.zh-CN.md)

**Extract the gyral net and its 3-hinge gyri from cortical surface meshes, driven by sulcal depth or by gradient-density maps.**

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![VTK](https://img.shields.io/badge/VTK-surfaces-1F6FEB)
[![paper](https://img.shields.io/badge/paper-GS--Net%20·%20ISBI%202026-B31B1B)](https://arxiv.org/abs/2601.08818)
[![homepage](https://img.shields.io/badge/homepage-ccao2679.com-555555)](https://www.ccao2679.com/research/#densitymap-gyralnet)

<img src="docs/figures/overview.png" alt="Overview poster: read VTK surface and sulcal depth, isolate gyral regions, skeletonize gyral crests, mark 3-hinge gyri" width="100%">

</div>

<!-- problem-framing:start -->

## Problem and evidence

> **Given the same cortical surface input, can completing broken gyral skeletons make the network's connectivity more consistent without adding links that the geometry does not support?**

**Why it matters:** Downstream graph analysis depends on stable nodes and edges. A broken skeleton or a wrong link changes the network's structure even when the visualisation still looks plausible.

**Central tension:** Repairing breaks raises connectivity, but it can also join structures that are not actually adjacent. A more complete graph is not necessarily a more accurate one.

[Full problem and verification plan](PROBLEM.md) · [中文](PROBLEM.zh.md)

<!-- problem-framing:end -->

## Highlights

- **What it builds:** the *gyral net* — the gyral crests traced over the cortical surface — and its **3-hinge gyri**, the three-way crest junctions that act as fine-scale cortical hubs.
- **In and out:** VTK surface and sphere meshes with per-vertex sulcal depth, curvature and cortical thickness in; gyral skeletons, 3-hinge vertices and graph files (VTK) out.
- **Why “DensityMap”:** besides the classic sulcal-depth route (`gyralnet.py`), `GDM_Net/` builds the same net from gradient-density maps on the 32k mesh, which is where the repository's name comes from.
- **Scale:** one run per subject, spread over CPU cores with GNU `parallel` (`3h_generate_zl.sh`); used on Human Connectome Project surfaces (164k fsaverage by default, 32k fs_LR in `GDM_Net/`). The Slurm scripts prepare surfaces and gradient features.

## Architecture

```mermaid
flowchart TD
    subgraph IN["Inputs per hemisphere"]
        MESH["sphere + inner surface<br/>*.withGrad.*.vtk"]
        MORPH["sulc · curv · thickness<br/>FreeSurfer morph files"]
    end

    GEN["32k_gradient_generate.py<br/>.grad, .rescale.grad"]

    subgraph SPLIT["Gyri / sulci split"]
        SULC["gyralnet.py<br/>sulc sign + curv fixes"]
        GDM["GDM_Net/gdm_net.py<br/>threshold gradient density"]
    end

    subgraph SKEL["Shared skeleton steps"]
        ISO["delete_isolated_point<br/>isolated-point cleanup"]
        THIN["thin gyri<br/>erode from margins"]
        TREE["create_tree<br/>grow by patch size"]
        TRIM["trim short branches<br/>reconnect breaks"]
        MISS["missing-link repair<br/>Dijkstra shortest paths"]
        HINGE["create_3hinge<br/>skeleton degree ≥ 3"]
    end

    OUT1["*_skelenton_allpoints_final.vtk<br/>sphere + surface"]
    OUT2["*_3hinge_vertex.vtk<br/>*_3hinge_ids.txt"]

    MESH --> SULC
    MORPH --> SULC
    MESH -->|gradient_density| GEN
    GEN --> GDM
    MESH --> GDM
    SULC --> ISO
    GDM --> ISO
    ISO --> THIN --> TREE --> TRIM --> MISS --> HINGE
    MISS --> OUT1
    HINGE --> OUT2
```

Read top to bottom: the two scripts in the middle differ in how they label each vertex gyral or sulcal (sulcal depth corrected by curvature in `gyralnet.py`, a thresholded gradient-density map in `GDM_Net/gdm_net.py`); from the isolated-point cleanup on, both run the same skeleton code, which `GDM_Net/utils/` repackages. Each step also writes intermediate VTK files (features, patch size, thinned gyri, initial tree and skeleton) next to the final outputs.

## How it works

| Step | What happens | Main functions (`gyralnet.py`) |
|---|---|---|
| 1. Read | Load the surface, sphere and per-vertex features; build vertex neighbourhoods | `read_vtk_file`, `get_connect_points` |
| 2. Isolate | Split gyri from sulci with sulcal depth and curvature, drop isolated vertices, find marginal points | `initialize_sulc_data`, `delete_isolated_point`, `find_marginal_point` |
| 3. Skeletonize | Thin each gyral region from its margins, grow a skeleton tree from the vertex with the largest patch, trim short branches, then repair missing crest links along Dijkstra shortest paths | `create_tree`, `find_skelenton`, `create_connection_for_missing_gyri` |
| 4. Mark | Write skeletons and 3-hinge vertices back to VTK on the sphere and the surface | `write_skelenton_by_connectionPair`, `draw_3h_sphere_single_vtk.py` |

## Quick start

```bash
pip install vtk nibabel numpy networkx
python3 gyralnet.py --root_dir="$PWD" \
    --subject_list_start_id=<id> --subject_list_end_id=<id+1> \
    --input_dir="<id>_recon" --out_dir=gyralnet_island_164k_flip
```

## Repository layout

| Path | Purpose |
|---|---|
| `gyralnet.py`, `gyralnet_test2.py` | Gyral-net extraction for a range of subjects |
| `utils/` | Neighbourhoods, sulcal data, trees, featured spheres, VTK helpers |
| `GDM_Net/` | Gradient-density-map variant of the pipeline and its utilities |
| `20240707_prepare_hcp_surface.py`, `20240608_prepare_hcp_fmri.py` | HCP surface and fMRI preparation |
| `32k_gradient_generate.py` | Gradient features on the 32k mesh |
| `draw_3h_sphere_single_vtk.py` | Draw 3-hinge points on a sphere or surface VTK |
| `3h_generate_zl.sh` | Run `gyralnet.py` for every subject folder with GNU `parallel` |
| `sbatch_*.py` | Slurm jobs for surface preparation and gradient features |

## Related papers

- C. Cao\*, T. Chen\*, N. Zhao\*, et al. *Gyral-Sulcal-Net: An Integrated Network Representation of Brain Folding Patterns.* IEEE ISBI 2026. [arXiv:2601.08818](https://arxiv.org/abs/2601.08818)
- C. Cao\*, J. Cheng\*, et al. *Deep-GyralNet: Enabling Machine Learning in Gyral Folding Pattern Extraction on Cortical Surface.* IEEE ISBI 2026. [doi:10.1109/ISBI61048.2026.11515596](https://doi.org/10.1109/ISBI61048.2026.11515596)
- C. Cao, X. Yu, L. Zhang, et al. *Enhancing Group-Wise Consistency in 3-Hinge Gyrus Matching via Anatomical Embedding and Structural Connectivity Optimization.* IEEE ISBI 2024. [doi:10.1109/ISBI56570.2024.10635893](https://doi.org/10.1109/ISBI56570.2024.10635893)

All papers, with figures and short summaries: [ccao2679.com/publications](https://www.ccao2679.com/publications/)

<details>
<summary>原始笔记（中文）</summary>

`tmp.py` 是一个处理大脑皮层表面数据的脚本，包含了多个功能模块。下面是对其主要功能和结构的概述：

### 主要功能

1. **读取和解析VTK文件**: 
    - `read_vtk_file(vtk_file)`: 读取VTK格式的文件并解析成VTK PolyData对象。
  
2. **获取点的连接信息**: 
    - `get_connect_points(surf_polydata)`: 获取每个点的邻居点。
    - `get_connect_points_gyri_part(surf_polydata, sulc_data)`: 获取指定皮层沟数据中的点的连接信息。
  
3. **孤立点处理**:
    - `delete_isolated_point(point_num, point_neighbor_points_dict, sulc_data)`: 删除皮层数据中孤立的点。

4. **边缘点查找**:
    - `find_marginal_point(points_list, point_neighbor_points_dict, sulc_data)`: 查找边缘点以及其在沟和凸起部分的位置。

5. **导出特征化的球面文件**:
    - `featured_sphere(orig_sphere_polydata, feature_file_dict, output)`： 根据特征数据导出特征化的球面文件。
  
6. **骨架树创建和连接**:
    - `create_tree(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, thin_sulc_data, output_prefix)`: 创建骨架树。
    - `find_skelenton(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, curv_data_delete_thicknessZero, original_sulc_data, thin_sulc_data, point_neighbor_points_dict, point_connect_points_dict_thin_gyri_parts, connected_lines_list, father_dict, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix)`: 通过创建并连接骨架树来查找骨架。

7. **寻找缺失的脑回连接并进行补全**:
    - `find_skelenton_missing(orig_sphere_polydata, orig_surf_polydata, skeleton_polydata, curv_data_delete_thicknessZero, original_sulc_data, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix)`: 找到缺失的脑回并补全骨架。

8. **其他工具函数**:
    - `draw_3hinge_on_surf(surf_polydata, hinge3_list, output_3hinge_vertex)`: 在表面上绘制3-hinge点。
    - `create_shortest_path(adj, start, end)`: 利用Dijkstra算法创建最短路径。
    - `main(args)`: 脚本的主函数，负责调用上述方法执行具体的功能。

### 总结

- **依赖库**： 包括 `vtk`, `nibabel`, `numpy`, `networkx`, `argparse` 等。
- **数据处理**： 脚本主要是读取、处理脑的3D结构数据，并进行骨架化处理。
- **功能模块**： 脚本清晰地将各个功能模块分开，使得每个函数只负责特定的任务。

---

```python
python3 ./gyralnet_test2.py
```


### RAW
```python
python3 /home/lab/Documents/HCP_data_32k_flip/gyralnet.py --root_dir="$PWD" --subject_list_start_id="$subject_id" --subject_list_end_id="$((subject_id + 1))" --input_dir="${subject_id}_recon" --out_dir='gyralnet_island_32k_flip'
```


---
### 20240608_prepare_hcp_fmri.py

vtk + fmri utils


---
DGM_0.7

```python
python3 GDM_0.7.py --root_dir=. --subject_list_start_id=100206 
```

</details>
