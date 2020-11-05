from audhelper.read import normalized_read



if __name__ == "__main__":
  aud = normalized_read('demo.wav')

  print(aud.shape)




