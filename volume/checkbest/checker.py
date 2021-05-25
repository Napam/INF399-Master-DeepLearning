import numpy as np 
import pandas as pd 

if __name__ == '__main__':
    df_concat = pd.read_csv('checkbest/losses_concat.csv')
    df_regular = pd.read_csv('checkbest/losses_regular.csv')
    df_splitfc = pd.read_csv('checkbest/losses_splitfc.csv')

    print("",df_concat.train.min())
    print("",df_regular.train.min())
    print("",df_splitfc.train.min())