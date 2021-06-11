import numpy as np 
import pandas as pd 





if __name__ == '__main__':
    df = pd.read_csv('mapdir/nogit_val_labels.csv')
    print(df.min())
    print(df.max())
    print(abs(df.max() - df.min()))

    # Xtest, diff
    # imgnr     999.000
    # class_      5.000
    # x           1.999
    # y           2.000
    # z           1.997
    # w           0.831
    # l           3.315
    # h           1.066
    # rx          0.999
    # ry          0.999
    # rz          1.000
    # dtype: float64