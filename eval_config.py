"""
This code is part of an adaptation/modification from the original project available at:
https://github.com/peterwang512/CNNDetection

The original code was created by Wang et al. and is used here under the terms of the license
specified in the original project's repository. Any use of this adapted/modified code
must respect the terms of such license.

Adaptations and modifications made by: Daniel Cabanas Gonzalez
Modification date: 08/04/2024
"""

from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = './dataset/test/'

# list of synthesis algorithms
vals = ['own', 'text2img', 'insight', 'inpainting']
# vals = ['text2img', 'insight']
# indicates if corresponding testset has multiple classes
multiclass = [0,0,0,0]

# model
model_path = 'weights/inpainting50.pth'
