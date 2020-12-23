import os
from os.path import join, split, basename

from glob import glob
import math
import wave

import random

import soundfile as sf
from samplerate import resample
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve

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
from audiomentations.core.utils import calculate_rms, calculate_desired_noise_rms

class MyGain(Gain):
  def apply(self, samples, sample_rate):
    return np.clip(samples * self.parameters["amplitude_ratio"], -1, 1)


class MyCompose:
  def __init__(self, transforms, p=1.0, max_augs=3):
    assert len(transforms), "No transforms provided"
    assert max_augs <= len(transforms), "max augs is smaller than transforms"
    self.transforms = transforms
    self.p = p
    self.max_augs = max_augs
    
    name_list = []
    for transform in transforms:
      name_list.append(type(transform).__name__)
    self.__name__ = '_'.join(name_list)

  def __call__(self, samples, sample_rate):
    if random.random() < self.p:
      picked_trans = random.sample(self.transforms, self.max_augs)
      
      for trans in picked_trans:
        samples = trans(samples, sample_rate)

    return samples


class MyAddImpulseResponse:
  def __init__(self, ir_path, p=1.0):
    self.ir_path = ir_path
    self.ir_files = glob(join(self.ir_path, '*.wav'))
    self.p = p

  def __call__(self, samples, sample_rate):
    if random.random() < self.p:
      ir, _ = sf.read(random.choice(self.ir_files))

      signal_ir = convolve(samples, ir)[:len(samples)]

      normalize_factor = max(np.max(np.abs(signal_ir)), 1e-6)
      signal_ir /= normalize_factor

      return signal_ir
    else:
      return samples


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
    MyAddImpulseResponse(p=_p, ir_path='data/impulse'),
    AddBackgroundNoise(sounds_path=sounds_path, p=_p),
    TimeStretch(p=_p/10),
    PitchShift(p=_p/30),
  ]
  
  return Compose(transforms, p=0.4, shuffle=True)


def compose_without_noise(ir_path='data/impulse'):
  _p = 0.25

  transforms = [
    AddGaussianNoise(p=_p),
    Shift(p=_p, min_fraction=-0.2, max_fraction=0.2),
    FrequencyMask(p=_p),
    TimeMask(p=_p, max_band_part=0.25),
    AddGaussianSNR(p=_p),
    ClippingDistortion(p=_p, max_percentile_threshold=20),
    MyAddImpulseResponse(p=_p, ir_path=ir_path),
    TimeStretch(p=_p/10),
    PitchShift(p=_p/25),
  ]
  
  return MyCompose(transforms, p=1.0, max_augs=3)


def nread(data, samples, sample_rate, shuffle, aug=None):
  # a faster implementation of normalize read with augmentation
  if isinstance(data, str):
    aud, sr = sf.read(data)
  elif isinstance(data, np.ndarray):
    aud = data.copy()
    sr = sample_rate
  else:
    raise Exception('Wrong read format! Only supporting filename/numpy.ndarray.')
  
  # reduce to mono channel
  if aud.ndim > 1: aud = aud.mean(axis=1)

  # augmentation
  if aug:
    aud = aug(aud, sr)

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


def add_background_noise(aud, noise_aud, min_snr_in_db=3, max_snr_in_db=30):
  snr_in_db = random.uniform(min_snr_in_db, max_snr_in_db)
  # snr_in_db = np.exp(random.uniform(np.log(min_snr_in_db), np.log(max_snr_in_db)))

  clean_rms = calculate_rms(aud)
  noise_rms = calculate_rms(noise_aud) + 1e-6
  desired_noise_rms = calculate_desired_noise_rms(clean_rms, snr_in_db)

  noise_aud = noise_aud * desired_noise_rms / noise_rms
  _aud = aud + noise_aud

  # normalize
  normalize_factor = max(1e-6, np.abs(_aud).max())
  _aud /= normalize_factor

  return _aud


def add_impulse_noise(aud, ir):  
  signal_ir = convolve(aud, ir)

  start = np.random.randint(len(ir)//2)
  signal_ir = signal_ir[start:start+len(aud)]

  normalize_factor = max(np.max(np.abs(signal_ir)), 1e-6)
  signal_ir /= normalize_factor

  return signal_ir


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



# ReadLargeWav          8.1966 s/100 epochs
# ReadWav               6.8874 s/100 epochs
class ReadLargeWav():
  def __init__(self, file_name):
    self.open(file_name)
  
  def open(self, file_name):
    self.f = wave.open(file_name, "rb")
    params = self.f.getparams()

    self.channels, self.sampwidth, self.framerate, self.nframes = params[:4]
    self.first_read   = True
    self.last_frames  = None
    self.duration     = self.nframes / self.channels / self.framerate

    print('Reading %s with %d...' % (file_name, self.duration))
  
  def read(self, time_duration = 1500, over_slide_time = 500):
    # unit ms
    if self.first_read:
      need_frames_count = int(time_duration * self.framerate / 1000)

      frames  = self.f.readframes(need_frames_count)

      if len(frames) != int(need_frames_count * self.sampwidth):
        print("read frames: " + str(len(frames)) + " " + str(need_frames_count * self.sampwidth))

        return None

      self.last_frames = frames            
      self.first_read  = False

      return frames
    else:
      need_frames_count = over_slide_time * self.framerate / 1000
      frames = self.f.readframes(int(need_frames_count))

      if len(frames) != int(need_frames_count * self.sampwidth):
        self.close()

        return None

      need_frames = self.last_frames + frames
      need_bytes_count = int(time_duration * self.framerate * self.sampwidth /1000)
      
      self.last_frames = need_frames[len(need_frames)-need_bytes_count:]
      
      return self.last_frames

  def close(self):
    self.f.close()


class StreamWav:
  def __init__(self, file_name, window=1500, stride=500):
    self.f = wave.open(file_name, 'rb')
    params = self.f.getparams()

    self.channels, self.sampwidth, self.framerate, self.nframes = params[:4]
    self.duration = self.nframes / self.channels / self.framerate
    self.window_frames = int(window * self.framerate / 1000)
    self.stride_frames = int(stride * self.framerate / 1000)

    assert self.nframes >= self.window_frames, 'Reading file too small or window too large, use other method!'

    self.overlap_frames = self.window_frames - self.stride_frames
    self.total_reads = math.floor((self.nframes - self.overlap_frames) / self.stride_frames)
    self.curr_read = 0
    self.last_frames = self.f.readframes(self.overlap_frames)

    print('Reading %s with %.2f s...' % (file_name, self.duration))

  def read(self):
    if self.curr_read == self.total_reads:
      self.f.close()  # close the file

      return self.curr_read, None  # indicating end of file

    read_frames = self.f.readframes(self.stride_frames)
    frames = self.last_frames + read_frames

    self.last_frames = frames[-self.overlap_frames*self.sampwidth:]
    self.curr_read += 1

    return self.curr_read-1, frames


if __name__ == "__main__":
  aug = compose('/home/ubuntu/Datasets/NLP/px/background1')

  import datetime

  st = datetime.datetime.now()
  for _ in range(1000):
    aud = nread('demo.wav', 24000, 16000, False, aug=aug)

  ed = datetime.datetime.now()

  print(ed - st)