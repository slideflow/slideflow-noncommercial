![slideflow logo](https://github.com/jamesdolezal/slideflow/raw/master/docs-source/pytorch_sphinx_theme/images/slideflow-banner.png)

[![Python application](https://github.com/slideflow/slideflow-gpl/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/slideflow/slideflow-gpl/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/slideflow-gpl.svg)](https://badge.fury.io/py/slideflow-gpl)
| [ArXiv](https://arxiv.org/abs/2304.04142) | [Docs](https://slideflow.dev) | [Cite](#reference)


**Slideflow-NonCommercial brings additional digital pathology deep learning tools to Slideflow, under the CC BY-NC 4.0 license.**

Slideflow is designed to provide an accessible, easy-to-use interface for developing state-of-the-art pathology models. While the core Slideflow package integrates with a wide range of cutting-edge methods and models, the variability in licensing practices necessitates that some functionality is distributed through separate add-on packages. **Slideflow-NonCommercial** extends Slideflow with additional tools available under the CC BY-NC 4.0 non-commercial license, ensuring that the core package remains as open and permissive as possible.

## Requirements
- Python >= 3.8
- [Slideflow](https://github.com/jamesdolezal/slideflow) >= 3.0
- [PyTorch](https://pytorch.org/) >= 1.12

## Installation
Slideflow-NonCommercial is easily installed via PyPI and will automatically integrate with Slideflow.

```
pip install slideflow-noncommercial
```

## Features
- **HistoSSL**, a pretrained foundation model ([GitHub](https://github.com/owkin/HistoSSLscaling) | [Paper](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2.full.pdf))
- **PLIP**, a pretrained foundation model ([GitHub](https://github.com/PathologyFoundation/plip) | [Paper](https://www.nature.com/articles/s41591-023-02504-3))
- **GigaPath**, a pretrained whole-slide foundation model ([GitHub](https://github.com/prov-gigapath/prov-gigapath) | [Paper](https://aka.ms/gigapath))
- **UNI**, a pretrained foundation model ([GitHub](https://github.com/mahmoodlab/UNI) | [Paper](https://www.nature.com/articles/s41591-024-02857-3))
- **BISCUIT**, an uncertainty quantification and thresholding algorithm ([GitHub](https://github.com/slideflow/biscuit) | [Paper](https://www.nature.com/articles/s41467-022-34025-x))
- **StyleGAN3**, a generative adversarial network (GAN) used for both image synthesis and [model explainability](https://www.nature.com/articles/s41698-023-00399-4) ([GitHub](https://github.com/NVlabs/stylegan3) | [Paper](https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf))

#### Foundation models

These foundation models are accessible using the [same interface](https://slideflow.dev/mil/#generating-features) all pretrained extractors utilize in Slideflow.

```python
import slideflow as sf

retccl = sf.build_feature_extractor('uni')
```

Please see the [Slideflow documentation](https://slideflow.dev/mil/#generating-features) for additional information on how feature extractors can be deployed and used. 

#### StyleGAN3

GANs can be trained and deployed in Slideflow, both programmatically and with [Slideflow Studio](https://slideflow.dev/studio/). Please see the [Slideflow docs](https://slideflow.dev/stylegan) for examples and instructions for use.

#### BISCUIT

The uncertainty quantification and thresholding algorithm BISCUIT will be automatically added as a submodule at `slideflow.biscuit`. Please see the [BISCUIT docs](https://github.com/slideflow/biscuit) for examples and use.


## License
This code is made available under the CC BY-NC 4.0 license for non-commercial research applications.

## Reference
If you find our work useful for your research, or if you use parts of this code, please consider citing as follows:

Dolezal, J.M., Kochanny, S., Dyer, E. et al. Slideflow: deep learning for digital histopathology with real-time whole-slide visualization. BMC Bioinformatics 25, 134 (2024). https://doi.org/10.1186/s12859-024-05758-x

```
@Article{Dolezal2024,
    author={Dolezal, James M. and Kochanny, Sara and Dyer, Emma and Ramesh, Siddhi and Srisuwananukorn, Andrew and Sacco, Matteo and Howard, Frederick M. and Li, Anran and Mohan, Prajval and Pearson, Alexander T.},
    title={Slideflow: deep learning for digital histopathology with real-time whole-slide visualization},
    journal={BMC Bioinformatics},
    year={2024},
    month={Mar},
    day={27},
    volume={25},
    number={1},
    pages={134},
    doi={10.1186/s12859-024-05758-x},
    url={https://doi.org/10.1186/s12859-024-05758-x}
}
```
