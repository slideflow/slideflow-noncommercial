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

from transformers import AutoImageProcessor, AutoModel
from slideflow.model import torch_utils
from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
import torch

# -----------------------------------------------------------------------------

class PhikonV2ImageFeatures(torch.nn.Module):

    def __init__(self,device='cuda'):
        super().__init__()
        self._model = AutoModel.from_pretrained("owkin/phikon-v2")
        self.device = device

    def forward(self, image_batch):
        self._model = self._model.half() #PhikonV2 is working with fp16
        self._model = self._model.to(self.device)
        image_batch = image_batch.to(self.device)
        with torch.inference_mode():
            outputs = self._model(**image_batch)
            features = outputs.last_hidden_state[:, 0, :]
            return features

def processor(image):
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    inputs = processor(image, return_tensors="pt")
    return inputs


class PhikonV2Features(TorchFeatureExtractor):
    """Phikon-v2 is a Vision Transformer Large pre-trained with Dinov2 self-supervised method 
    on PANCAN-XL, a dataset of 450M 20x magnification histology images sampled from 60K whole 
    slide images. PANCAN-XL only incorporates publicly available datasets: CPTAC (6,193 WSI) 
    and TCGA (29,502 WSI) for malignant tissue, and GTEx for normal tissue (13,302 WSI).

    Phikon-v2 improves upon Phikon, our previous fondation model pre-trained with iBOT on 40M 
    histology images from TCGA (6k WSI), on a large variety of weakly-supervised tasks tailored 
    for biomarker discovery. Phikon-v2 is evaluated on external cohorts to avoid any data 
    contamination with PANCAN-XL pre-training dataset, and benchmarked against an exhaustive 
    panel of representation learning and foundation models. 
    
    Feature dimensions: 1024

    Hugging Face: https://huggingface.co/owkin/phikon-v2

    """
    tag = 'phikonv2'
    license = """Special license (non-commercial use only). License available at https://github.com/owkin/phikon-v2"""
    citation = """
@misc{filiot2024phikonv2largepublicfeature,
      title={Phikon-v2, A large and public feature extractor for biomarker prediction}, 
      author={Alexandre Filiot and Paul Jacob and Alice Mac Kain and Charlie Saillard},
      year={2024},
      eprint={2409.09173},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2409.09173}, 
}
"""

    def __init__(self, device='cuda', **kwargs):
        super().__init__(**kwargs)

        self.device = torch_utils.get_device(device)
        self.model = PhikonV2ImageFeatures(device=self.device)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 1024
        self.transform = processor
        self.preprocess_kwargs = dict(standardize=False)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.phikonv2.PhikonV2Features',
        )
