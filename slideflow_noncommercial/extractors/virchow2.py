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
import torch
from timm.layers import SwiGLUPacked

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class Virchow2Features(TorchFeatureExtractor):
    """Virchow2 pretrained feature extractor.
    The feature extractor is a Vision Transformer (ViT) model pretrained on a
    3.1M whole-slide dataset of histopathology images, trained on different magnifications
    (5x, 10x, 20x, 40x). Virchow2 is built and distributed by Paige, and is available on 
    Hugging Face at hf-hub:paige-ai/Virchow2.
    The transformer outputs both a class token (size: 1280) and 5 patch token (256 x 1280),
    wherein the tokens 1-4 are register tokens, so they are ignored.
    As recommended by the authors, the final downstream feature vector is a concatenation
    of the class token and an average pool of the 5th patch token, resulting in a final
    vector size of 2560.
    Feature dimensions: 2560
    Manuscript: Zimmermann, E., et al. (2024). Scaling Self-Supervised Mixed Magnification 
    Models in Pathology. arXiv preprint arXiv:2408.00738 (2024).
    Hugging Face: https://huggingface.co/paige-ai/Virchow2
    """

    tag = 'virchow2'
    license = """CC-BY-NC-ND 4.0 (non-commercial use only). Please see the original license at https://huggingface.co/paige-ai/Virchow2."""
    citation = """
@misc{zimmermann2024virchow2,
      title={Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology}, 
      author={Eric Zimmermann and Eugene Vorontsov and Julian Viret and Adam Casson and Michal Zelechowski and George Shaikovski and Neil Tenenholtz and James Hall and Thomas Fuchs and Nicolo Fusi and Siqi Liu and Kristen Severson},
      year={2024},
      eprint={2408.00738},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2408.00738},
}
"""

    def __init__(self, weights, device='cuda', **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.model = timm.create_model(
            "vit_huge_patch14_224",
            img_size=224,
            patch_size=14,
            init_values=1e-5,
            num_classes=0,
            reg_tokens=4,
            mlp_ratio=5.3375,
            global_pool="",
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        td = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(td, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 2560

        # Note that Virchow2 uses bicubic interpolation
        # https://huggingface.co/paige-ai/Virchow2/blob/main/config.json
        self.transform = self.build_transform(img_size=224, interpolation='bicubic')
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = weights

    def _process_output(self, output):
        """Concatenate class and patch tokens into a single embedding."""
        class_token = output[:, 0]   # 1 x 1280
        patch_tokens = output[:, 5:] # 1 x 261 x 1280 -> 1 x 256 x 1280
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # 1 x 2560
        return embedding.to(torch.float32)

    def dump_config(self):
        """Return a dictionary of configuration parameters.
        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.
        """
        return self._dump_config(
            class_name='slideflow.model.extractors.virchow2.Virchow2Features',
            weights=self._weights
        )