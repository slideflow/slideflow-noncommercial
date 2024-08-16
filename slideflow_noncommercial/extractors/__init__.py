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

from slideflow import log
from slideflow.model.extractors import register_torch

# -----------------------------------------------------------------------------

@register_torch
def phikon(**kwargs):
    from .phikon import PhikonFeatures
    return PhikonFeatures(**kwargs)

@register_torch
def histossl(**kwargs):
    log.warning("'histossl' has been renamed to 'phikon', in accordance with the author's naming.")
    return phikon(**kwargs)

@register_torch
def plip(**kwargs):
    from .plip import PLIPFeatures
    return PLIPFeatures(**kwargs)

@register_torch
def uni(weights, **kwargs):
    from .uni import UNIFeatures
    return UNIFeatures(weights, **kwargs)

@register_torch("gigapath")
def gigapath(**kwargs):
    from .gigapath import GigapathFeatures
    return GigapathFeatures(**kwargs)

@register_torch("gigapath.slide")
def gigapath(**kwargs):
    from .gigapath import GigapathSlideFeatures
    return GigapathSlideFeatures(**kwargs)

@register_torch("gigapath.tile")
def gigapath(**kwargs):
    from .gigapath import GigapathTileFeatures
    return GigapathTileFeatures(**kwargs)
