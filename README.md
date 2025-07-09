<!--
 * @Author: TMJ
 * @Date: 2025-07-08 22:23:12
 * @LastEditors: TMJ
 * @LastEditTime: 2025-07-09 14:51:31
 * @Description: 请填写简介
-->
# *N,N'*-Dioxide/metal complex catalyzed Michael Addiditon: Data & ML

## Environments

We recommend using the miniforge to create a conda environment for this project.

### main environment

```bash
conda create -n nn-dioxide python=3.11 -y
conda activate nn-dioxide
pip install .
```

### LocalMapper environment

We use the LocalMapper to build Atom-Atom mapping. But this module is only availbale in python <= 3.7. Thus we need to create a separate environment for LocalMapper.

```bash
conda create -n localmapper python=3.7 -y # name of environment must be "localmapper"
pip install localmapper
```
