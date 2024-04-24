"""
This code is part of an adaptation/modification from the original project available at:
https://github.com/peterwang512/CNNDetection

The original code was created by Wang et al. and is used here under the terms of the license
specified in the original project's repository. Any use of this adapted/modified code
must respect the terms of such license.

Adaptations and modifications made by: Daniel Cabanas Gonzalez
Modification date: 08/04/2024
"""

import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *


def main():
    # Running tests
    opt = TestOptions().parse(print_options=False)
    model_name = os.path.basename(model_path).replace('.pth', '')
    rows = [["{} model testing on...".format(model_name)],
            ['testset', 'accuracy', 'avg precision']]

    print("{} model testing on...".format(model_name))
    for v_id, val in enumerate(vals):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = True    # testing without resizing by default

        model = resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model.cuda()
        model.eval()

        acc, ap, _, _, _, _ = validate(model, opt)
        rows.append([val, acc, ap])
        print("({}) acc: {}; ap: {}".format(val, acc, ap))

    csv_name = results_dir + '/{}.csv'.format(model_name)
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)

if __name__ == '__main__':
    main()