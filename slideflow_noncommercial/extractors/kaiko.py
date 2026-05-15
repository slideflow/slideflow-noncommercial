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

import torch
from slideflow.model import torch_utils
from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class KaikoFeatures(TorchFeatureExtractor):
    """Kaiko ViT-B/8 pretrained feature extractor.

    Feature dimensions: 384

    Manuscript: https://arxiv.org/abs/2404.15217

    Github: https://github.com/kaiko-ai/towards_large_pathology_fms

    """

    tag = 'kaiko'
    license = """Custom License (non-commercial use only). Please see the original license at https://github.com/kaiko-ai/towards_large_pathology_fms/blob/main/LICENSE."""
    citation = """
@misc{ai2024largescale,
    title={Towards Large-Scale Training of Pathology Foundation Models}, 
    author={kaiko.ai and Nanne Aben and Edwin D. de Jong and Ioannis Gatopoulos and Nicolas Känzig and Mikhail Karasikov and Axel Lagré and Roman Moser and Joost van Doorn and Fei Tang},
    year={2024},
    eprint={2404.15217},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
"""

    def __init__(self, device='cuda', **kwargs):
        super().__init__(**kwargs)

        self.device = torch_utils.get_device(device)
        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb8", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 384
        self.transform = self.build_transform(img_size=224, center_crop=True, norm_mean=[0.5, 0.5, 0.5],norm_std=[0.5, 0.5, 0.5])
        self.preprocess_kwargs = dict(standardize=False)


    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.kaiko.KaikoFeatures',
        )