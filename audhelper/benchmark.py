
import pandas as pd
import numpy as np

def pformat(idx, cols, tk, hs):
  ''' cols stands for a list of strings '''
  print(str(idx).ljust(hs), end=tk)
  for _i, _a in enumerate(cols):
    print(_a, end=tk if _i != len(cols) - 1 else '\n')

def pbenchmark(table, alias=None, project=None):
  _t = table.pivot_table(
    index='word', columns='alias', values='score',
    fill_value=0, aggfunc=lambda x: np.round(np.mean(x), 2)
  )

  token = '\t| '
  head_size = ((max(len(project), 16) - 1) // 8 + 1) * 8 - 1
  alias = _t.columns if alias is None else alias

  print(f'Benchmark prroject {project}:')
  pformat(project, alias, token, head_size)
  pformat('-'*(head_size+1), ['-'*7]*len(alias), '|', head_size)
  for _w in _t.index:
    msg = [_t[_a].loc[_w] if _a in _t.columns else '-' for _a in alias]
    pformat(_w, msg, token, head_size)
    

if __name__ == "__main__":
  pass