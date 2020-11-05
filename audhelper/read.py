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


if __name__ == "__main__":
  pass