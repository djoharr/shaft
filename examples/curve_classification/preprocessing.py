import numpy as np
import pandas as pd
import os
from os.path import join, dirname

def preprocess_csv(path, sep=','):
    df = pd.read_csv(path, sep=sep)
    idxs = np.unique(df.indexNumber)
    all_curves = []
    cols = list(set(df.columns) - set(['X', 'Y', 'pointOrder']))
    clean_df = pd.DataFrame(columns=cols)
    for idx in idxs:
        el = df[df.indexNumber == idx]
        curve_points = np.stack((el.Y, el.X), axis=1)
        clean_df.loc[idx] = el[cols].iloc[0]
        all_curves.append(curve_points)

    clean_df = clean_df.set_index('indexNumber')
    clean_df.index.rename('idx', inplace=True)

    return np.array(all_curves), clean_df


def main():
    data_path = join(dirname(__file__), '../../data/curves')

    leaves_path = join(data_path, 'original_data', 'leaves.csv')
    leaves_curves, leaves_info = preprocess_csv(leaves_path)
    leaves_dir = join(data_path, 'leaves')
    os.mkdir(leaves_dir)
    leaves_info.to_csv(join(leaves_dir,'info.csv'))
    np.save(join(leaves_dir, 'curves.npy'), leaves_curves)


    shells_path = join(data_path, 'original_data', 'shells.csv')
    shells_curves, shells_info = preprocess_csv(shells_path)
    shells_dir = join(data_path, 'shells')
    os.mkdir(shells_dir)
    shells_info.to_csv(join(shells_dir,'info.csv'))
    np.save(join(shells_dir, 'curves.npy'), shells_curves)


    vases_path = join(data_path, 'original_data', 'vases.csv')
    vases_curves, vases_info = preprocess_csv(vases_path)
    vases_dir = join(data_path, 'vases')
    os.mkdir(vases_dir)
    vases_info.to_csv(join(vases_dir,'info.csv'))
    np.save(join(vases_dir, 'curves.npy'), vases_curves)


if __name__ == '__main__':
    main()
