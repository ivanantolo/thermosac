import os
import glob
import pandas as pd

# directory containing your .csv files
path = 'dsp'

# build a list of all CSV filepaths
all_files = glob.glob(os.path.join(path, '*.csv'))

# read each file into a DataFrame, then concatenate them all
df_list = [pd.read_csv(f, sep=';') for f in all_files]
big_df = pd.concat(df_list, ignore_index=True)

big_df.to_csv('dsp.csv', sep=';', index=False)
