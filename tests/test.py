from audhelper.read import normalized_read
from audhelper.micro import DummyModel, kws_monitor



if __name__ == "__main__":
  aud = normalized_read('demo.wav', target_sample_rate=16000, target_samples=32000, shuffle=False)

  print(aud.shape)

  prj = 'MOB' 
  interval = 100.0
  duration = 3000.0
  sr = 16000
  sp = 24000
  gamma = 0.3
  words = ['negative', 'jiuming', 'baojing']
  model = DummyModel(len(words))

  kws_monitor(model, prj, interval, duration, sr, sp, words, gamma)




