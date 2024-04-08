"""
This code is part of an adaptation/modification from the original project available at:
https://github.com/peterwang512/CNNDetection

The original code was created by Wang et al. and is used here under the terms of the license
specified in the original project's repository. Any use of this adapted/modified code
must respect the terms of such license.

Adaptations and modifications made by: Daniel Cabanas Gonzalez
Modification date: 08/04/2024
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser
