import pandas as pd
import numpy as np

np.set_printoptions(threshold=np.nan)
pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 70)

def thing():
  with pd.option_context('display.max_rows', 999, 'display.max_columns', 3):
    print df