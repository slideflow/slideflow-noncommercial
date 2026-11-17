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

# H-optimus normalization, per the official H-optimus-0 model card.
# https://huggingface.co/bioptimus/H-optimus-0
_HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


class H0MiniFeatures(TorchFeatureExtractor):
    """H0-mini pretrained feature extractor.

    H0-mini is a lightweight, distilled variant of H-optimus-0 (bioptimus), a
    Vision Transformer pretrained by self-supervision on histopathology
    whole-slide images. Where H-optimus-0 is a ViT-giant, H0-mini is distilled
    down to a ViT-Base backbone (embed_dim 768, depth 12, patch 14, 4 register
    tokens, SwiGLU-packed MLP, LayerScale) for faster inference and triage while
    retaining the parent model's representation.

    The transformer emits 261 tokens for a 224x224 input: index 0 is the class
    token, indices 1-4 are register tokens (discarded), and indices 5+ are the
    256 patch tokens. Following H-optimus-0's documented downstream convention,
    the default embedding is the class token alone (768-dim). Passing
    ``pool='cls_mean'`` instead concatenates the class token with the mean of the
    patch tokens (Virchow2 style), giving a 1536-dim vector.

    The SwiGLU-packed MLP and SiLU activation are load-bearing: the pretrained
    checkpoint uses a gated MLP, so ``timm.create_model`` must be given
    ``mlp_layer=SwiGLUPacked`` and ``act_layer=torch.nn.SiLU`` (with
    ``mlp_ratio=4096/768`` to size the packed hidden dimension) or the weights
    fail to load.

    Feature dimensions: 768 (default) or 1536 (``pool='cls_mean'``)

    Hugging Face (parent model): https://huggingface.co/bioptimus/H-optimus-0

    """

    tag = 'h0_mini'
    weights_hash = '39bf4bcc791036011cee29fd0513a8fee6f240c8c650a865d9a363b1c5231b56'
    license = """Non-commercial use only. Please refer to the original authors (bioptimus, H-optimus-0): https://huggingface.co/bioptimus/H-optimus-0."""
    citation = """
@software{hoptimus0,
  author={Saillard, Charlie and Jenatton, Rodolphe and Llinares-L\\'opez, Felipe and Mariet, Zelda and Cahan\\'e, David and Durand, Eric and Vert, Jean-Philippe},
  title={H-optimus-0},
  year={2024},
  url={https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0},
}
"""

    def __init__(
        self,
        weights: str,
        pool: str = 'cls',
        device: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        if pool not in ('cls', 'cls_mean'):
            raise ValueError(
                f"Invalid pool '{pool}'; expected 'cls' or 'cls_mean'.")

        if version.parse(timm.__version__) < version.parse('1.0.0'):
            log.warning(
                "H0-mini requires timm version 1.0.0 or later. Please update "
                "timm with `pip install --upgrade timm`.")

        self.device = torch_utils.get_device(device)
        # mlp_ratio/mlp_layer/act_layer are load-bearing (see class docstring).
        self.model = timm.create_model(
            "vit_base_patch14_reg4_dinov2",
            pretrained=False,
            img_size=224,
            num_classes=0,
            init_values=1e-5,
            mlp_ratio=4096/768,
            global_pool="",
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        td = torch.load(weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict(td, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self._pool = pool
        self.num_features = 768 if pool == 'cls' else 1536

        # H-optimus uses bicubic interpolation and its own normalization.
        self.transform = self.build_transform(
            img_size=224,
            interpolation='bicubic',
            norm_mean=_HOPTIMUS_MEAN,
            norm_std=_HOPTIMUS_STD,
        )
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = weights

    def _process_output(self, output: torch.Tensor) -> torch.Tensor:
        """Reduce the token sequence to a tile embedding.

        H0-mini emits 261 tokens: index 0 is the class token, indices 1-4 are
        register tokens (discarded), and indices 5+ are the 256 patch tokens.

        Args:
            output (torch.Tensor): Raw transformer output, shape (B, 261, 768).

        Returns:
            torch.Tensor: Tile embedding, shape (B, 768) for ``pool='cls'`` or
            (B, 1536) for ``pool='cls_mean'``.
        """
        class_token = output[:, 0]  # (B, 768)
        if self._pool == 'cls':
            embedding = class_token
        else:
            patch_tokens = output[:, 5:]  # (B, 256, 768); tokens 1-4 are registers
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # (B, 1536)
        return embedding.to(torch.float32)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.h0_mini.H0MiniFeatures',
            weights=self._weights,
            pool=self._pool,
        )
