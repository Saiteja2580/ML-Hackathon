import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE,ADASYN

def smote(x,y):
    smt = SMOTE(random_state=42)
    x_balanced,y_balanced = smt.fit_resample(x,y)
    return [x_balanced,y_balanced]

def ada_smote(x,y):
    adasyn = ADASYN(random_state=42)
    x_balanced,y_balanced = adasyn.fit_resample(x,y)
    return [x_balanced,y_balanced]
