from glob import glob
from os.path import join, split

import random
import numpy as np


from .read import nread, compose

def checkdata(datasets):
  for _i in datasets:
    for _d in _i['train']:
      assert glob(join(_d, '*.wav')), f'Empty dir {_d}!'

  
def datasets_from_cfg(cfg_model, cfg_dataset):
  checkdata(cfg_dataset)

  train_dirs = []
  val_dirs = []

  bs = cfg_model['batch_size']
  sr = cfg_model['sample_rate']
  sp = cfg_model['samples']

  for _dataset in cfg_dataset:
    if isinstance(_dataset['train'], list):
      train_dirs += [(_dataset['word'], _) for _ in _dataset['train']]
    
    if isinstance(_dataset['val'], list):
      val_dirs += [(_dataset['word'], _) for _ in _dataset['val']]

  print(train_dirs)
  print(val_dirs)

  augs = compose(cfg_model['augmentation']) if 'augmentation' in cfg_model else None

  _train = AudDataset(train_dirs, batch_size=bs, samples=sp, sample_rate=sr, training=True, augs=augs)
  _val = AudDataset(val_dirs, batch_size=bs, samples=sp, sample_rate=sr, training=False, augs=None)

  return _train, _val

class AudDataset(object):
  def __init__(self, dirs, batch_size, samples, sample_rate, training, augs=None):
    """
    prepare data, we are not suppose to change training state
    """
    self.__data = []
    self.__words_list = ['_unknown_', 'jiuming', 'baojing', 'qiangjie', 'sharen']

    dirs = [dirs] if not isinstance(dirs, list) else dirs

    for _word, _dir in dirs:
      audios = glob(join(_dir, '*.wav'))

      for audio in audios:
        self.__data.append(
          (self.__words_list.index(_word), audio)
        )
    
    self.__start = 0
    random.shuffle(self.__data)
    self.__batch_size = batch_size
    self.__samples = samples
    self.__sample_rate = sample_rate
    self.__training = training
    self.__steps = (len(self.__data) - 1) // self.__batch_size + 1
    self.__augs = augs

  def __len__(self):
    return self.__steps

  @property
  def size(self):
    return len(self.__data)

  def __iter__(self):
    self.__start = 0
    random.shuffle(self.__data)

    return self

  def __next__(self):
    if self.__start < self.__steps:
      batch_data = self.__data[
        self.__start * self.__batch_size: min(self.size, (self.__start + 1) * self.__batch_size)
      ]
      self.__start += 1

      cnt = len(batch_data)
      auds = np.empty((cnt, self.__samples))
      labels = []

      for i, (label, audio) in enumerate(batch_data):
        _aud = nread(audio, self.__samples, self.__sample_rate, self.__training, self.__augs)
        if self.__augs and random.random() < 0.1:
          _aud = _aud[::-1]
          label = 0
        auds[i, :] = _aud
        labels.append(label)
      
      return auds, labels
    else:
      raise StopIteration()

  def eval(self, model):
    # make sure the model is totally constracted with current dataset
    assert self.__sample_rate == model.sample_rate, "Different sample rate!"
    assert self.__samples == model.samples, "Different samples"
      
