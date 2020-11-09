import numpy as np

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


def report(res, c, s):
  """report alarms from result returned from stream_test"""
  reports = np.argmax(res, axis=1)
  scores = np.max(res, axis=1)

  pre_s = 0
  pre_r = 0
  cum = 0
  cums = []
  alarms = []

  for s, r in zip(scores, reports):
    if r != 0 and r == pre_r:
      if cum == c:
        alarms.append(1)
        cum = -s
      else:
        alarms.append(0)
        cum += 1
    else:
      alarms.append(0)
      cum = max(0, cum-1)

    pre_r, pre_s = r, s
    cums.append(cum)

  return alarms, cums







if __name__ == "__main__":
  pass