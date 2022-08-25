# protocolBLAST
This repository provides data and tools to compare sample preparation protocols for correlative multimodal imaging experiments in life sciences.

The data and code in this repository complement the findings described in [this paper](https://doi.org/10.3389/fcell.2022.880696). 

## About the paper:
We tested a variety of sample preparation protocols (SPPs) for volume electron microscopy and analysed the occurrences of artefacts with respect to SPP variables. We also devised a method to quantify dataset registration (warping) accuracy. Both advances help increase the efficiency of studying the brain using a correlative multimodal imaging approach that we developed and described in detail in [this previous study](https://www.nature.com/articles/s41467-022-30199-6).

## Content of this repo:
This repo provides:
- All compiled data and scripts used for the analyses of SPP artefacts, so that anyone interested may regenerate plots or run their own analysis.
- Links to datasets and annotations used to quantify warping accuracy, so that anyone interested may examine the warpings or run their own analysis. The warping toolbox is available at [warpAnnotations](https://github.com/FrancisCrickInstitute/warpAnnotations).

## Installing this repo:
You can explore the 3D datasets in [webKnossos](https://webknossos.org/) following the `wk_scene` links. No need to install anything. 

Clone this repository to your preferred location. 

Install [jupyter](https://jupyter.org/install) to run some analyses.

Install the [warpAnnotations](https://github.com/FrancisCrickInstitute/warpAnnotations) toolbox to warp and analyse correlative multimodal annotations.

## Usage: revisiting artefact analysis:
The jupyter notebook provided ([here](https://github.com/FrancisCrickInstitute/protocolBLAST/tree/main/1-analysis)) loads all data tables and reproduces the plots shown in the publication. 

You can find examples of all artefact types reported below:
| artefact name | sample ID | batch ID | LXRT dataset link| 
| -------- | -------- | -------- | -------- |
| perfect 1st slab | Y129 | CLEM210308 | [wk_scene](https://wklink.org/4525) |
| perfect 2nd slab | Y132 | CLEM210308 | [wk_scene](https://wklink.org/9035) |
| sideways | Y193 | PIP210913 | [wk_scene](https://wklink.org/6754) |
| crack | Y137 | CLEM210308 | [wk_scene](https://wklink.org/5768) |
| undefined | C376 | CLEM171127 | [wk_scene](https://wklink.org/3522) |
| murky | C332 | CLEM170809 | [wk_scene](https://wklink.org/8644) |
| smoky | C430 | CLEM180205 | [wk_scene](https://wklink.org/5062) |
| patchy | Y151 | CLEM210308 | [wk_scene](https://wklink.org/1534) |
| overstain | Y257 | SBS211128 | [wk_scene](https://wklink.org/2971) |
| central | Y081 | PIP201014 | [wk_scene](https://wklink.org/1925) |
| bubble | Y054 | PIP200921 | [wk_scene](https://wklink.org/3835) |
| spotty | Y139 | CLEM210308 | [wk_scene](https://wklink.org/8480) |
| necrotic | Y127 | CLEM210308 | [wk_scene](https://wklink.org/9399) |
| sandy | Y169 | CLEM210810 | [wk_scene](https://wklink.org/9182) |
| sample with 2 artefacts | Y233 | PIP211019 | [wk_scene](https://wklink.org/5027) |
| sample with 2 artefacts | Y028 | PIP200909 | [wk_scene](https://wklink.org/2266) |
| sample with 3 artefacts | Y052 | PIP200921 | [wk_scene](https://wklink.org/6084) |

## Usage: revisiting warping analysis:
The following scenes will let you visit datasets of the same tissue region in the mouse brain acquired with in-vivo 2-photon, SXRT and SBEM. In each of those scenes, you will find traces generated delineating the same blood vessel across all imaging modalities. 

| Content | 2P dataset link | SXRT dataset link | SBEM dataset link |
| --------| --------------- | ----------------- | ----------------- |
| blood vessel tracings used to quantify warping accuray (Fig. 6c),<br />example soma and its surrounding blood vessels, correlated among 3 datasets by warping (Fig. 6h) | [wk_scene](https://wklink.org/4525) | [wk_scene](https://wklink.org/6137) | [wk_scene](https://wklink.org/6224) |

In order to further annotate, warp and analyse annotation in those (or new) datasets, install the [warpAnnotations](https://github.com/FrancisCrickInstitute/warpAnnotations) toolbox.

## Questions and feedback
If you have any questions, please contact us: [Yuxin Zhang](mailto:yuxin.zhang@crick.ac.uk), [Carles Bosch](mailto:carles.bosch@crick.ac.uk).





