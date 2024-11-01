import pandas as pd
import sys


def load_csv(filepath):
    Mutations_df = pd.read_csv(filepath)
    return Mutations_df
                       
if __name__ == '__main__':
    data_path = sys.argv[1]
    output_file = sys.argv[2]
    data = load_csv(data_path)
    data.to_csv(output_file, index=False)