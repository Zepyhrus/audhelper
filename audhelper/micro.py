import argparse
import queue
import sys
from functools import partial

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

from read import normalized_read

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
  
  preds = params['model'].infer(normalized_read(params['plotdata'][-sp:, 0], sp, sr, shuffle=False))
  params['resdata'] = np.roll(params['resdata'], -1, axis=0)
  params['resdata'][-1, :] = preds * gamma + params['resdata'][-2, :] * (1 - gamma)
  
  params['lines1'][0].set_ydata(params['plotdata'][:, 0])
  for i, line2 in enumerate(params['lines2']):
    line2.set_ydata(params['resdata'][:, i])

  return params['lines1'] + params['lines2']

def kws_monitor(model, interval, duration, samplerate, samples, num_labels, gamma):
  length = int(duration * sr / 1000)
  steps = int(duration / interval)

  # initialize core
  core = {
    'plotdata': np.zeros((length, 1)),
    'resdata': np.zeros((steps, num_labels)),
    'model': model,
    'queue': queue.Queue(),
    'gamma': gamma
  }
  # initialize figure
  fig, (ax1, ax2) = plt.subplots(2, 1)

  ax1.axis((0, length-1, -1, 1))
  ax1.set_yticks([0])
  ax1.yaxis.grid(True)
  ax1.tick_params(bottom='off', top='off', labelbottom='off',
                  right='off', left='off', labelleft='off')

  ax2.legend(['negative', 'jiuming', 'baojing'], loc='upper left')
  ax2.axis((0, steps-1, -0.05, 1.05))

  fig.tight_layout(pad=0)

  core['lines1'] = ax1.plot(core['plotdata'])
  core['lines2'] = ax2.plot(core['resdata'])

  
  # stream plot
  stream = sd.InputStream(
    device=None, channels=1, samplerate=samplerate, callback=partial(__callback__, params=core)
  )
  ani = FuncAnimation(fig, partial(__update__, params=core), interval=interval, blit=True)

  with stream:
    plt.show()



if __name__ == "__main__":  
  interval = 100.0
  duration = 3000.0
  sr = 16000
  sp = 24000
  nl = 3
  gamma = 0.3
  model = DummyModel(nl)

  kws_monitor(model, interval, duration, sr, sp, nl, gamma)

  



