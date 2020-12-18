import threading, queue, time

import numpy as np
from scipy.optimize import linear_sum_assignment as KM

import soundfile as sf

from .textgrid import TextGrid
from .read import nread, StreamWav


def textgrid_res(grid_file):
  text = open(grid_file).read()

  fid = TextGrid(text)
  
  for i, tier in enumerate(fid):
    res = tier.simple_transcript


  label_st = []
  label_ed = []
  labels = []
  for _st, _ed, _wd in res:
    _st = float(_st)
    _ed = float(_ed)

    label_st.append(_st)
    label_ed.append(_ed)
    labels.append(_wd)

  return labels, label_st, label_ed


def report(res, _c, _s, gamma):
  """report alarms from result returned from stream_test"""
  _res = res.copy()
  # smooth
  for _, _scores in enumerate(_res):
    if _ == 0: continue

    _res[_, :] = _res[_, :] * gamma + _res[_-1, :] * (1 - gamma)

  reports = np.argmax(_res, axis=1)
  scores = np.max(_res, axis=1)

  pre_r = 0
  cum = 0
  cums = []
  alarms = []

  for r in reports:
    alarm = [0] * _res.shape[1]
    if r != 0 and (r == pre_r or _c == 0):
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
      diff = abs(t1[i] - t2[j])
      cost_matrix[i, j] = 1 / diff

  gt_idx, dt_idx = KM(cost_matrix, maximize=True)

  correct = 0
  for _g, _d in zip(gt_idx, dt_idx):
    if abs(t1[_g] - t2[_d]) <= interval:
      correct += 1
  
  return correct


def report_from_res(res_file, grid_file, interval=500, method='f1', word_index=1):
  if grid_file is not None:
    # load labels and file
    words, st, et = textgrid_res(grid_file)
    st, et = np.array(st), np.array(et)
    labels = np.zeros(len(words))

    for i, word in enumerate(words):
      if '救命' in word:
        label = 1
      elif '报警' in word:
        label = 2
      elif '抢劫' in word:
        label = 3
      elif '杀人' in word:
        label = 4
      else:
        label = 0
      
      labels[i] = label


  # load result
  res = np.genfromtxt(res_file, delimiter=',')
  res = np.concatenate((res[[0], :], res[1::int(interval/100), ])) # interval for 200 ms
  best = 5
  _s = min(20, int(2000 / interval))

  for _g in np.arange(1, 11) / 10:
    for _c in range(11):
      alarms, _ = report(res, _c, _s, _g)
      
      alarms = alarms[:, word_index]
      time_alarms = np.arange(alarms.shape[0]) * interval / 1000

      if grid_file is None:
        continue
      
      t1 = st[labels == word_index]
      t2 = time_alarms[alarms == word_index]

      correct = alarm_eval(t1, t2, 2)
      recall = correct / (len(t1) + 1e-12) * 100
      precis = correct / (len(t2) + 1e-12) * 100
      f1 = 2 * precis * recall / (precis + recall + 1e-12)

      if method == 'f1':
        target = f1
      elif method == 'recall':
        target = recall
      elif method == 'precision':
        target = precis
      else:
        raise Exception(f'Wrong method {method} must be in f1/recall/precision!')

      if target > best - 5:
        if target > best:
          best = target
        print(f'word {word_index} on {method} - recall: {recall:.2f}, precision: {precis:.2f}, f1: {f1:.2f} @ {_c}, {_g}, {len(t1)} labels, {len(t2) - correct} false alarms.')


class WavQueue(queue.Queue):
  """
  The WavQueue holds the same use of Pipeline, but around 12% faster than Pipeline
    WavQueue: 6.7s on room.wav, it is more than 70 times faster than stream_test
    Pipeline: 7.8s on room.wav
    stream_test: 9 min 18.1s on room.wav
  Increasing batch size from 256 to 512, will slightly faciliate the speed, 6.7s to 6.5s
  """
  def __init__(self, model):
    self.model = model
    self.batch_size = self.model.batch_size
    self.lock = threading.Lock()
    super().__init__(maxsize=2) # 2 batch is enough
  
  def eat(self, file_name, res_file=None):
    self.res_file = res_file
    self.wav_file = file_name
    self.wav = StreamWav(file_name=file_name, window=1500, stride=100)
    self.res = np.zeros((self.wav.total_reads + 1, self.model.num_classes), dtype=np.float32)
    self.res[0, 0] = 1
    self.already_get = 0
    self.batches = 0
    self.total_batches = (self.wav.total_reads - 1) // self.batch_size + 1

  def consume(self):
    while True:
      batches, batch_aud = self.get()

      preds = self.model.batch_infer(batch_aud)

      st = (batches - 1) * self.batch_size + 1
      ed = st + len(batch_aud)

      self.res[st:ed, :] = preds

      if batches % 50 == 0: # usually 25 batch would be a fine update interval
        print('%.2f finished on %s batches...' % (batches / self.total_batches, self.total_batches))

      if batches == self.total_batches: break
    
    if self.res_file:
      print('Res file saved to %s' % self.res_file)
      np.savetxt(self.res_file, self.res, delimiter=',')

  def produce(self):
    while True:
      batch_idx = 0
      batch_raw_data = []

      # now we fetch raw data from wav file
      with self.lock: # using lock to make sure all raw_data are read continously
        while batch_idx < self.batch_size and self.already_get < self.wav.total_reads:
          _, raw_data = self.wav.read()

          if raw_data is None:
            break
          else:
            batch_raw_data.append(raw_data)
          
          batch_idx += 1
          self.already_get += 1
      
      # now we construct a batch of aud data
      batch_aud = np.empty((batch_idx, self.model.samples), dtype=np.float32)

      for i, _raw in enumerate(batch_raw_data):
        batch_aud[i] = nread(
          np.frombuffer(_raw, dtype=np.int16) / 32768.,
          self.model.samples,
          self.model.sample_rate,
          shuffle=False,
          aug=None
        )
      
      self.batches += 1

      self.put((self.batches, batch_aud))
      if self.batches == self.total_batches: break



if __name__ == "__main__":
  pass