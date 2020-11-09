import numpy as np
from scipy.optimize import linear_sum_assignment as KM

import soundfile as sf



def stream_test(f, model, gamma, interval):
  samples = model.samples
  samplerate = model.sample_rate
  overlap = samples - int(interval / 1000 * samplerate)
  steps = int(samples / samplerate * 1000 / interval)

  frames = sf.blocks(f, blocksize=samples, overlap=overlap)

  res = [np.zeros((1, model.num_classes))]

  for frame in frames:
    preds = model.infer(frame)
    res.append(preds * gamma + res[-1] * (1 - gamma))

  return np.concatenate(res)


def report(res, _c, _s):
  """report alarms from result returned from stream_test"""
  reports = np.argmax(res, axis=1)
  scores = np.max(res, axis=1)

  pre_r = 0
  cum = 0
  cums = []
  alarms = []

  for r in reports:
    alarm = [0] * res.shape[1]
    if r != 0 and r == pre_r:
      if cum == _c:
        alarm[r] = 1
        cum = -_s
      else:
        alarm[0] = 1
        cum += 1
    else:
      alarm[0] = 1
      cum = max(0, cum-1)

    alarms.append(alarm)
    cums.append(cum)
    pre_r = r
  
  alarms = np.array(alarms)
  cums = np.array(cums)

  return alarms, cums


def alarm_eval(t1, t2, interval):
  cost_matrix = np.zeros((len(t1), len(t2)))

  for i in range(len(t1)):
    for j in range(len(t2)):
      diff = t1[i] - t2[j]
      cost_matrix[i, j] = 1 / diff

  gt_idx, dt_idx = KM(cost_matrix, maximize=True)

  correct = 0
  for _g, _d in zip(gt_idx, dt_idx):
    if t1[_g] - t2[_d] <= interval:
      correct += 1
  
  return correct





if __name__ == "__main__":
  pass