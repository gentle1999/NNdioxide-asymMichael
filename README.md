<!--
 * @Author: TMJ
 * @Date: 2025-07-08 22:23:12
 * @LastEditors: TMJ
 * @LastEditTime: 2025-11-11 21:18:40
 * @Description: 请填写简介
-->

# _N,N'_-Dioxide/metal complex catalyzed Michael Addiditon: Data & ML

## Environments

We recommend using the miniforge to create a conda environment for this project.

### main environment

```bash
conda create -n nn-dioxide python=3.11 -y
conda activate nn-dioxide
pip install . -e
```

### LocalMapper environment

We use the LocalMapper to build Atom-Atom mapping. But this module is only availbale in python <= 3.7. Thus we need to create a separate environment for LocalMapper.

```bash
conda create -n localmapper python=3.7 -y # name of environment must be "localmapper"
conda activate localmapper
pip install localmapper rdkit
```

## Notebooks

Follow the instructions to run the experiments.

## Cite this work

```bibtex
@article{tangDatadrivenModelingNN2025,
  title = {Data-driven Modeling of {{{\emph{N}}}}{\emph{,}}{{{\emph{N}}}}{\emph{{$\prime$}}} -dioxide/Metal-catalyzed Asymmetric Michael Additions},
  author = {Tang, Miao-Jiong and Zhang, Tinghui and Huang, Qiuhao and Li, Shuwen and Liu, Rui and Li, Hongye and Chen, Xiaofan and Dong, Shunxi and Liu, Xiaohua and Feng, Xiaoming and Hong, Xin},
  year = 2025,
  month = nov,
  journal = {Angewandte Chemie International Edition},
  pages = {e18560},
  issn = {1433-7851, 1521-3773},
  doi = {10.1002/anie.202518560},
  langid = {english},
}
```
