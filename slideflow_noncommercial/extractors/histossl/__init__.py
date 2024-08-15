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

# This file incorporates work from HistoSSLscaling, which is licensed
# under a  non-commercial license. The original license and source code 
# can be found at https://github.com/owkin/HistoSSLscaling.


"""HistoSSL Pretrained model.

Model (iBOTViT) and pretrained weights are provided by Owkin, under the
license found in the LICENSE file in the same directory as this source file.

"""
import os
import gdown
import slideflow as sf
from slideflow.util import make_cache_dir_path

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

from .ibotvit import iBOTViT


# -----------------------------------------------------------------------------

class HistoSSLFeatures(TorchFeatureExtractor):
    """
    HistoSSL pretrained feature extractor.
    Feature dimensions: 768
    GitHub: https://github.com/owkin/HistoSSLscaling
    """

    tag = 'histossl'
    url = 'https://drive.google.com/uc?id=1uxsoNVhQFoIDxb4RYIiOtk044s6TTQXY'
    license = """
This model is developed and licensed by Owkin, Inc. The license for use is
provided in the LICENSE file in the same directory as this source file
(slideflow/model/extractors/histossl/LICENSE), and is also available
at https://github.com/owkin/HistoSSLscaling. By using this feature extractor,
you agree to the terms of the license.
"""
    citation = """
@article{Filiot2023ScalingSSLforHistoWithMIM,
	author       = {Alexandre Filiot and Ridouane Ghermi and Antoine Olivier and Paul Jacob and Lucas Fidon and Alice Mac Kain and Charlie Saillard and Jean-Baptiste Schiratti},
	title        = {Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling},
	elocation-id = {2023.07.21.23292757},
	year         = {2023},
	doi          = {10.1101/2023.07.21.23292757},
	publisher    = {Cold Spring Harbor Laboratory Press},
	url          = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757},
	eprint       = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757.full.pdf},
	journal      = {medRxiv}
}
"""
    MD5 = 'e7124eefc87fe6069bf4b864f9ed298c'

    def __init__(self, device=None, weights=None, **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.print_license()
        if weights is None:
            weights = self.download()
        self.device = torch_utils.get_device(device)
        self.model = iBOTViT(
            architecture='vit_base_pancan',
            encoder='student',
            weights_path=weights
        )
        self.model.to(self.device)

        # ---------------------------------------------------------------------
        self.num_features = 768
        self.transform = self.build_transform(img_size=224)
        self.preprocess_kwargs = dict(standardize=False)
        # ---------------------------------------------------------------------

    @staticmethod
    def download():
        """Download the pretrained model."""
        dest = make_cache_dir_path('histossl')
        dest = os.path.join(dest, 'ibot_vit_base_pancan.pth')
        if not os.path.exists(dest):
            gdown.download(HistoSSLFeatures.url, dest, quiet=False)
        if sf.util.md5(dest) != HistoSSLFeatures.MD5:
            raise sf.errors.ChecksumError(
                f"Downloaded weights at {dest} failed MD5 checksum."
            )
        return dest

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name=f'slideflow.model.extractors.histossl.HistoSSLFeatures',
        )
