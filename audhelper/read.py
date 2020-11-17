import os
import wave

import soundfile as sf
from samplerate import resample
import numpy as np
from scipy.io import wavfile

from audiomentations import (
  Compose,
  Gain,                   #  0.1701   0.1086
  AddGaussianNoise,       #  1.0177   0.7528
  TimeStretch,            # 10.8556   8.4630
  PitchShift,             # 29.8609  23.1933
  Shift,                  #  0.1838   0.1261
  AddImpulseResponse,     #  3.3720   1.8873
  FrequencyMask,          #  2.4121   1.8783
  TimeMask,               #  0.1652   0.1133
  AddGaussianSNR,         #  1.1160   0.8297
  ClippingDistortion,     #  0.5428   0.4300
  AddBackgroundNoise,     #  1.7820   1.7376
  AddShortNoises          #  1.6541   1.6303
)


#@title Helper functions and classes
def normalized_read(filename, target_samples, target_sample_rate, shuffle):
  """Reads and normalizes a wavfile."""
  if isinstance(filename, str):
    # _, data = wavfile.read(filename)
    data, sr = sf.read(filename)
  elif isinstance(filename, np.ndarray):
    data = filename
    sr = target_sample_rate
  else:
    raise Exception('Wrong read format! Only supporting filename/binary/numpy.ndarray.')

  if data.ndim > 1:
    data = data.mean(axis=1)

  if sr != target_sample_rate:
    data = resample(data, target_sample_rate/sr)
  # crop
  if data.shape[0] > target_samples:
    if shuffle:
      idx = np.random.randint(data.shape[0] - target_samples)
    else:
      idx = (data.shape[0] - target_samples) // 2
    data = data[idx:(idx+target_samples)]

  # pad
  if data.shape[0] < target_samples:
    if shuffle:
      pad_left = np.random.randint(target_samples - data.shape[0])
    else:
      pad_left = (target_samples - data.shape[0]) // 2
    pad_right = target_samples - data.shape[0] - pad_left

    if pad_left > 0 or pad_right > 0:
      data = np.pad(data, (pad_left, pad_right), 'constant')

  samples_99_percentile = np.percentile(np.abs(data), 99.9)
  normalized_samples = data / (samples_99_percentile + 1e-6)
  normalized_samples = np.clip(normalized_samples, -1, 1)

  return normalized_samples

class MyGain(Gain):
  def apply(self, samples, sample_rate):
    return np.clip(samples * self.parameters["amplitude_ratio"], -1, 1)


def compose(sounds_path):
  _p = 0.2

  transforms = [
    MyGain(p=_p),
    AddGaussianNoise(p=_p),
    Shift(p=_p, min_fraction=-0.25, max_fraction=0.25),
    FrequencyMask(p=_p),
    TimeMask(p=_p, max_band_part=0.25),
    AddGaussianSNR(p=_p),
    ClippingDistortion(p=_p, max_percentile_threshold=20),
    AddBackgroundNoise(sounds_path='data/background1', p=_p),
    TimeStretch(p=_p/10),
    PitchShift(p=_p/30),
  ]
  
  return Compose(transforms, p=0.4, shuffle=True)


def nread(data, samples, sample_rate, shuffle, aug=None):
  # a faster implementation of normalize read with augmentation
  if isinstance(data, str):
    aud, sr = wavfile.read(data)
  elif isinstance(data, np.ndarray):
    aud = data.copy()
    sr = sample_rate
  else:
    raise Exception('Wrong read format! Only supporting filename/numpy.ndarray.')
  
  # reduce to mono channel
  if aud.ndim > 1: aud = aud.mean(axis=1)

  # augmentation
  if aug: aud = aug(aud, sr)

  # resample
  if sr != sample_rate: aud = resample(aud, sample_rate/sr)

  _size = aud.shape[0]
  # crop
  if _size > samples:
    idx = np.random.randint(_size - samples) if shuffle else (_size - samples) // 2
    aud = aud[idx:(idx+samples)]
  
  # pad
  if _size < samples:
    pad_left = np.random.randint(samples - _size) if shuffle else (samples - _size) // 2
    pad_right = samples - _size - pad_left

    aud = np.pad(aud, (pad_left, pad_right), 'constant')

  # normalize
  normalize_factor = max(1e-6, np.max(np.abs(aud)))
  aud /= normalize_factor

  return aud


def aread(audio, method='sf', aug=None):
  if method == 'sf':
    aud, sr = sf.read(audio)
  elif method == 'rosa':
    aud, sr = rosa.load(audio)
  elif method == 'dub':
    song = AudioSegment.from_wav(audio)
    aud = np.frombuffer(song.raw_data, dtype=np.short) / 32768
    sr = song.frame_rate
  elif method == 'sci':
    sr, raw = wavfile.read(audio)
    aud = raw / 32768
  else:
    raise Exception('Wrong audio read method!')

  return aud, sr


def awrite(audio, sample_rate, aud, method='sci'):
  if method == 'sci':
    wavfile.write(audio, sample_rate, (aud * 32768).astype(np.int16))
  else:
    raise Exception('Wrong audio read method!')


class ReadLargeWav():
  def __init__(self, file_name):
    self.open(file_name)
  
  def open(self, file_name):
    try:
      self.f = wave.open(file_name, "rb")
      params = self.f.getparams()

      self.channels, self.sampwidth, self.framerate, self.nframes = params[:4]
      self.open_flag   = True
      self.first_read  = True
      self.last_frames = None
      self.duration    = self.nframes / self.channels / self.framerate

      return True
    except Exception as err:
      print("open " + file_name + " fail")
      self.open_flag = False

      return False
  
  #unit ms
  def read(self,time_duration = 1500,over_slide_time = 500):
    if(self.first_read):
      need_frames_count = time_duration * self.framerate / 1000
      print(need_frames_count)
      frames  = self.f.readframes(int(need_frames_count))
      print(len(frames))
      if(len(frames) != int(need_frames_count *self.sampwidth) ):
        print ("read frames:"+str(len(frames)) + " " + str(need_frames_count *self.sampwidth))

        return None

      print(type(frames))
      self.last_frames = frames            
      self.first_read  = False

      return frames
    else:
      need_frames_count = over_slide_time * self.framerate / 1000
      frames = self.f.readframes(int(need_frames_count))

      if(len(frames) != int(need_frames_count *self.sampwidth) ):           
        return None

      need_frames = self.last_frames + frames
      need_bytes_count = int(time_duration * self.framerate * self.sampwidth /1000)
      
      self.last_frames = need_frames[len(need_frames)-need_bytes_count:]
      return self.last_frames

  def close(self):
    self.f.close()


if __name__ == "__main__":
  pass