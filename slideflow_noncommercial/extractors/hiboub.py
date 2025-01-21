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

class HibouBImageFeatures(torch.nn.Module):

    def __init__(self,device='cuda'):
        super().__init__()
        self._model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
        self.device = torch_utils.get_device(device)

    def forward(self, image_batch):
        #self._model = self._model.half()
        self._model = self._model.to(self.device)
        image_batch = image_batch.to(self.device)
        with torch.inference_mode():
            outputs = self._model(**image_batch)
            features = outputs.last_hidden_state[:, 0, :]
            return features

def processor(image):
    processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
    inputs = processor(image, return_tensors="pt")
    return inputs


class HibouBFeatures(TorchFeatureExtractor):
    """Hibou-B - a Foundational Vision Transformer for digital pathology pretrained on
     a private dataset using DINOv2 framework. 
    
    Feature dimensions: 1024

    Hugging Face: https://huggingface.co/histai/hibou-b

    """
    tag = 'hiboub'
    license = """Apache 2.0. License available at https://github.com/histai/hibou-b"""
    citation = """
@misc{nechaev2024hibou,
    title={Hibou: A Family of Foundational Vision Transformers for Pathology},
    author={Dmitry Nechaev and Alexey Pchelnikov and Ekaterina Ivanova},
    year={2024},
    eprint={2406.05074},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
"""

    def __init__(self, device='cuda', **kwargs):
        super().__init__(**kwargs)

        self.device = torch_utils.get_device(device)
        self.model = HibouBImageFeatures(device=self.device)
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
            class_name='slideflow.model.extractors.hiboub.HibouBFeatures',
        )
