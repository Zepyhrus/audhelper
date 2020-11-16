from glob import glob
from os.path import join, split

import random
import numpy as np


from .read import normalized_read


class AudDataset(object):
  def __init__(self, dirs, batch_size, samples, sample_rate, training):
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
    self.__batch_size = batch_size
    self.__samples = samples
    self.__sample_rate = sample_rate
    self.__training = training
    self.__steps = (len(self.__data) - 1) // self.__batch_size + 1

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
      auds = np.zeros((cnt, self.__samples))
      labels = []

      for i, (label, audio) in enumerate(batch_data):
        auds[i, :] = normalized_read(audio, self.__samples, self.__sample_rate, self.__training)
        labels.append(label)
      
      return auds, labels
    else:
      raise StopIteration()

  def eval(self, model):
    # make sure the model is totally constracted with current dataset
    assert self.__sample_rate == model.sample_rate, "Different sample rate!"
    assert self.__samples == model.samples, "Different samples"
      