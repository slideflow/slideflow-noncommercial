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
#
# The H-optimus-1 model and weights have their OWN separate license:
# CC BY-NC-ND 4.0 (see https://huggingface.co/bioptimus/H-optimus-1).
# Copyright 2026 Bioptimus. All rights reserved.
# The model requires Hugging Face access approval and acceptance of its terms.


import timm
import torch

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor


# -----------------------------------------------------------------------------

class Hoptimus1Features(TorchFeatureExtractor):
    """H-optimus-1 pretrained feature extractor.

    This class is used to extract tile-level features from H-optimus-1.

    Feature dimensions: 1536

    Model: https://huggingface.co/bioptimus/H-optimus-1

    License: CC BY-NC-ND 4.0 (non-commercial research; model terms apply).
    """

    tag = 'hoptimus1'
    license = """CC BY-NC-ND 4.0 (model license: https://huggingface.co/bioptimus/H-optimus-1). Copyright 2026 Bioptimus. All rights reserved."""
    citation = """
@inproceedings{scalbert2026hoptimus1,
  title={H-optimus-1: A foundation model for computational histopathology},
  author={Scalbert, Marin and Saillard, Charlie and Peeters, Thomas and Gonzalez, Liam and Valter, Dasha and Llinares-López, Felipe and Mariet, Zelda E. and Jenatton, Rodolphe},
  booktitle={Proceedings of the American Association for Cancer Research Annual Meeting 2026; Part 2 (Late-Breaking, Clinical Trial, and Invited Abstracts)},
  volume={86},
  number={8\\_Suppl},
  pages={LB174},
  year={2026},
  publisher={AACR},
  journal={Cancer Research},
  doi={10.1158/1538-7445.AM2026-LB174}
}
"""

    def __init__(self, weights=None, device='cuda', **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=(weights is None),
            init_values=1e-5,
            dynamic_img_size=False
        )
        if weights is not None:
            td = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(td, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 1536
        self.transform = self.build_transform(norm_mean=(0.707223, 0.578729, 0.703617),norm_std=(0.211883, 0.230117, 0.177517))
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = weights

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.
        """
        return self._dump_config(
            class_name='slideflow.model.extractors.hoptimus1.Hoptimus1Features',
            weights=self._weights
        )

