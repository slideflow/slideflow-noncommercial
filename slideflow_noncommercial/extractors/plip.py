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
from torchvision import transforms
from typing import Union

try:
    from transformers import CLIPModel, CLIPProcessor
    from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
except ImportError:
    raise ImportError(
        "The PLIP feature extractor requires the 'transformers' package. "
        "You can install it with 'pip install transformers'."
    )

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class CLIPImageFeatures(torch.nn.Module):

    def __init__(self, weights='vinid/plip'):
        super().__init__()
        self._model = CLIPModel.from_pretrained(weights)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            x = self._model.get_image_features(x)
            return x[0]
        else:
            x = self._model.get_image_features(x)
            return x


class PLIPFeatures(TorchFeatureExtractor):
    """
    PLIP pretrained feature extractor.
    Feature dimensions: 512
    GitHub: https://github.com/PathologyFoundation/plip
    """

    tag = 'plip'
    license = "No license provided by the authors."
    citation = """
@article{huang2023visual,
    title={A visual--language foundation model for pathology image analysis using medical Twitter},
    author={Huang, Zhi and Bianchi, Federico and Yuksekgonul, Mert and Montine, Thomas J and Zou, James},
    journal={Nature Medicine},
    pages={1--10},
    year={2023},
    publisher={Nature Publishing Group US New York}
}
"""

    def __init__(self, device=None, center_crop=False, resize=False, **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        if center_crop and resize:
            raise ValueError("center_crop and resize cannot both be True.")

        self.device = torch_utils.get_device(device)
        self.model = CLIPImageFeatures("vinid/plip")
        self.model.eval()
        self.model.to(self.device)

        # ---------------------------------------------------------------------
        self.num_features = 512
        if center_crop:
            all_transforms = [transforms.CenterCrop(224)]
        elif resize:
            all_transforms = [transforms.Resize(224)]
        else:
            all_transforms = []
        all_transforms += [
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=OPENAI_CLIP_MEAN,
                std=OPENAI_CLIP_STD),
        ]
        self.transform = transforms.Compose(all_transforms)
        self.preprocess_kwargs = dict(standardize=False)
        self._center_crop = center_crop
        self._resize = resize
        # ---------------------------------------------------------------------


    def text_preprocess(self, x: str):
        if not hasattr(self, '_text_preprocess'):
            self._text_preprocess = CLIPProcessor.from_pretrained("vinid/plip")
        return self._text_preprocess(
            text=x,
            return_tensors='pt',
            max_length=77,
            padding="max_length",
            truncation=True
        )

    def get_text_features(self, x: Union[str, "torch.Tensor"]):
        if isinstance(x, str):
            x = self.text_preprocess(x)
        x = x.to(self.device)
        with torch.inference_mode():
            x = self.model._model.get_text_features(**x)
        return x


    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        cls_name = self.__class__.__name__
        return {
            'class': f'slideflow.model.extractors.plip.{cls_name}',
            'kwargs': {
                'center_crop': self._center_crop,
                'resize': self._resize
            },
        }
