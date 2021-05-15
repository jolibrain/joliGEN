import visdom
import json
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--loss_log_file_path', type=str, default='./', help='Path where losses.json is saved.')
parser.add_argument('--port', type=str, default='8097', help='Port used by wisdom.')
parser.add_argument('--env_name', type=str, default='losses', help='Visdom environment name.')

opt, _ = parser.parse_known_args()

path=os.path.join(opt.loss_log_file_path,'losses.json')
name=opt.env_name

f = open(path,)
  
losses = json.load(f)

f.close()

X=np.stack([np.array(losses['X'])] * len(losses['legend']), 1)
Y=np.array(losses['Y'])

vis = visdom.Visdom(port=opt.port,env=name)

vis.line(
    Y,
    X,
    opts={
        'title': ' loss over time',
        'legend': losses['legend'],
        'xlabel': 'epoch',
        'ylabel': 'loss'},
    win=0)
