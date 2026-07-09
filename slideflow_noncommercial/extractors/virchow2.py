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

from typing import Optional

import timm
import torch
from packaging import version
from timm.layers import SwiGLUPacked

from slideflow import log
from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class Virchow2Features(TorchFeatureExtractor):
    """Virchow2 pretrained feature extractor.

    Virchow2 is a Vision Transformer (ViT-H/14) pretrained on 3.1M whole-slide
    histopathology images across multiple magnifications (5x, 10x, 20x, 40x).
    It is built and distributed by Paige and is available on Hugging Face at
    hf-hub:paige-ai/Virchow2.

    The transformer emits a class token and 256 patch tokens (each 1280-dim),
    preceded by 4 register tokens that are discarded. As recommended by the
    authors, the downstream embedding concatenates the class token with the
    mean of the patch tokens, giving a 2560-dim feature vector.

    The SwiGLU MLP and SiLU activation are load-bearing: the pretrained
    checkpoint uses a gated MLP, so ``timm.create_model`` must be given
    ``mlp_layer=SwiGLUPacked`` and ``act_layer=torch.nn.SiLU`` or the weights
    fail to load.

    Feature dimensions: 2560

    Manuscript: Zimmermann, E., et al. (2024). Virchow2: Scaling
    Self-Supervised Mixed Magnification Models in Pathology. arXiv preprint
    arXiv:2408.00738.

    Hugging Face: https://huggingface.co/paige-ai/Virchow2

    """

    tag = 'virchow2'
    weights_hash = '16df265ffde657add04a6a3dfd60d8cc266ce0ed'
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

    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        if version.parse(timm.__version__) < version.parse('1.0.0'):
            log.warning(
                "Virchow2 requires timm version 1.0.0 or later. Please update "
                "timm with `pip install --upgrade timm`.")

        self.device = torch_utils.get_device(device)
        # Weights download from Hugging Face when no local checkpoint is given.
        # mlp_layer/act_layer are load-bearing (see class docstring).
        self.model = timm.create_model(
            "hf_hub:paige-ai/Virchow2",
            pretrained=(weights is None),
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        if weights is not None:
            td = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(td, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 2560

        # Note that Virchow2 uses bicubic interpolation.
        # https://huggingface.co/paige-ai/Virchow2/blob/main/config.json
        self.transform = self.build_transform(img_size=224, interpolation='bicubic')
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = weights

    def _process_output(self, output: torch.Tensor) -> torch.Tensor:
        """Concatenate the class token and mean-pooled patch tokens.

        Virchow2 emits 261 tokens: index 0 is the class token, indices 1-4 are
        register tokens (discarded), and indices 5+ are the 256 patch tokens.

        Args:
            output (torch.Tensor): Raw transformer output, shape (B, 261, 1280).

        Returns:
            torch.Tensor: Tile embedding, shape (B, 2560).
        """
        class_token = output[:, 0]    # (B, 1280)
        patch_tokens = output[:, 5:]  # (B, 256, 1280); tokens 1-4 are registers
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # (B, 2560)
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