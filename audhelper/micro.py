import os
from os.path import join, split

import queue
import sys
from functools import partial

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

import time
from datetime import date

class DummyModel(object):
  def __init__(self, nl):
    self.nl = nl

  def infer(self, samples):
    return np.random.randn(self.nl) / 2 + 0.5
# prepare function
def __callback__(indata, frames, time, status, params):
  """This is called (from a separate thread) for each audio block."""
  if status:
    print(status, file=sys.stderr)
  # Fancy indexing with mapping creates a (necessary!) copy:
  params['queue'].put(indata[:, [0]])

def __update__(frame, params):
  while True:
    try:
      data = params['queue'].get_nowait()
    except queue.Empty:
      break
    shift = len(data)
    params['plotdata'] = np.roll(params['plotdata'], -shift, axis=0)
    params['plotdata'][-shift:, :] = data
  
  preds = params['model'].infer(params['plotdata'][-params['samples']:, 0])
  params['resdata'] = np.roll(params['resdata'], -1, axis=0)
  params['resdata'][-1, :] = preds * params['gamma'] + params['resdata'][-2, :] * (1 - params['gamma'])

  if params['dst'] is not None:
    if np.argmax(params['resdata'][-2, :]) != 0:
      name = date.today().strftime('%Y-%m-%d') + '_' + time.strftime('%H-%M-%S', time.localtime()) + '.wav'
      
      sf.write(join(params['dst'], name), params['plotdata'][:, 0], params['model'].sample_rate)
  
  params['lines1'][0].set_ydata(params['plotdata'][:, 0])
  for i, line2 in enumerate(params['lines2']):
    line2.set_ydata(params['resdata'][:, i])

  return params['lines1'] + params['lines2']

def kws_monitor(model, project, interval, duration, gamma, dst=None):
  samples = model.samples
  samplerate = model.sample_rate
  length = int(duration * samplerate / 1000)
  steps = int(duration / interval)
  num_labels = model.num_classes
  words = model.wanted_words

  # initialize core
  if dst is not None:
    os.makedirs(dst, exist_ok=True)
  
  core = {
    'plotdata': np.zeros((length, 1)),
    'resdata': np.zeros((steps, num_labels)),
    'model': model,
    'queue': queue.Queue(),
    'gamma': gamma,
    'samplerate': samplerate,
    'samples': samples,
    'dst': dst
  }
  # initialize figure
  fig, (ax1, ax2) = plt.subplots(2, 1)

  core['lines1'] = ax1.plot(core['plotdata'])
  core['lines2'] = ax2.plot(core['resdata'])

  ax1.title.set_text(f'{project}-{gamma:.2f}')
  ax1.axis((0, length-1, -1, 1))
  ax1.set_yticks([0])
  ax1.yaxis.grid(True)
  ax1.tick_params(bottom='off', top='off', labelbottom='off',
                  right='off', left='off', labelleft='off')

  ax2.legend(words, loc='upper left')
  ax2.axis((0, steps-1, -0.05, 1.05))

  fig.tight_layout(pad=0)
  
  # stream plot
  stream = sd.InputStream(
    device=None, channels=1, samplerate=samplerate, callback=partial(__callback__, params=core)
  )
  ani = FuncAnimation(fig, partial(__update__, params=core), interval=interval, blit=True)

  with stream:
    plt.show()

if __name__ == "__main__":
  pass

  



