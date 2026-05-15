# Slideflow-NonCommercial - Add-ons for the deep learning library Slideflow
# Copyright (C) 2024 James Dolezal
#
# This file is part of Slideflow-NonCommercial.
#
# Slideflow-NonCommercial is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
#
# You are free to share, copy, and redistribute the material in any medium or format, and to adapt, remix, transform, and build upon the material, as long as you follow the terms of the license.
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
#
# Slideflow-NonCommercial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Creative Commons Attribution-NonCommercial 4.0 International License for more details.
#
# You should have received a copy of the Creative Commons Attribution-NonCommercial 4.0 International License
# along with Slideflow-NonCommercial. If not, see <https://creativecommons.org/licenses/by-nc/4.0/>.

import timm

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor




# -----------------------------------------------------------------------------

class DinoBloomFeatures(TorchFeatureExtractor):
    """DinoBloom pretrained feature extractor in 4 sizes (small, base, large, giant)

    This class is used to extract tile-level features from DinoBloom.

    Feature dimensions: 384,768,1024,1536

    Manuscript: https://arxiv.org/abs/2404.05022

    Github: https://github.com/marrlab/DinoBloom

    """
    tag = 'dinobloom'
    license = """Apache License 2.0 (License available at https://github.com/marrlab/DinoBloom)"""
    citation = """
@misc{dinobloom,
      title={DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology}, 
      author={Valentin Koch and Sophia J. Wagner and Salome Kazeminia and Ece Sancar and Matthias Hehr and Julia Schnabel and Tingying Peng and Carsten Marr},
      year={2024},
      eprint={2404.05022},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

    def __init__(self, model_size='small', device='cuda', **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.model = timm.create_model(
            f"hf-hub:1aurent/vit_{model_size}_patch14_224.dinobloom",
            pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        feature_sizes={"small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536}
        self.num_features = int(feature_sizes[model_size])
        self.transform = self.build_transform(img_size=518,center_crop=True, interpolation= 'bicubic',norm_mean=[0.485, 0.456, 0.406],norm_std=[0.229, 0.224, 0.225])
        self.preprocess_kwargs = dict(standardize=False)


    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.dinobloom.DinoBloomFeatures',
        )
    