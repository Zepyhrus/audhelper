import os
import wave

import soundfile as sf
from samplerate import resample
import numpy as np


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
  normalized_samples = data / (samples_99_percentile + 1e-12)
  normalized_samples = np.clip(normalized_samples, -1, 1)

  return normalized_samples


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