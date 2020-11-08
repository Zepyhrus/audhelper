import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from glob import glob
import argparse
import yaml
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import soundfile as sf





def __update__(frame, params):
  preds = params['model'].infer(frame)

  params['resdata'] = np.roll(params['resdata'], -1, axis=0)
  params['resdata'][-1, :] = preds * params['gamma'] + params['resdata'][-2, :] * (1 - params['gamma'])
  
  params['lines1'][0].set_ydata(frame)
  for i, line2 in enumerate(params['lines2']):
    line2.set_ydata(params['resdata'][:, i])

  return params['lines1'] + params['lines2']
  

def stream_test(f, model, gamma, interval, show=False):
  samples = model.samples
  samplerate = model.sample_rate
  overlap = samples - int((1000 - interval) / 1000 * samplerate)
  steps = int(samples / samplerate * 1000 / interval)

  frames = sf.blocks(f, blocksize=samples, overlap=overlap)

  core = {
    'resdata': np.zeros((steps, model.num_classes)),
    'model': model,
    'gamma': gamma,
    'samplerate': model.sample_rate,
    'samples': model.samples
  }

  if show:
    fig, (ax1, ax2) = plt.subplots(2, 1)
  
    ax1.axis((0, samples-1, -1, 1))
    ax2.axis((0, steps-1, -0.1, 1.1))

    core['lines1'] = ax1.plot(np.zeros(model.samples))
    core['lines2'] = ax2.plot(core['resdata'])


    fig.tight_layout(pad=0)

    try:
      ani = FuncAnimation(fig, partial(__update__, params=core), frames=frames, interval=interval)
      plt.show()
    except:
      pass
  else:
    for frame in frames:
      preds = core['model'].infer(frame)
      core['resdata'] = np.roll(core['resdata'], -1, axis=0)
      core['resdata'][-1, :] = preds * core['gamma'] + core['resdata'][-2, :] * (1 - core['gamma'])

  return core['resdata']

if __name__ == "__main__":
  pass