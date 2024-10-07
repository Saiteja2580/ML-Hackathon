from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



def train_logistic_model(x,y):
    

    model = LogisticRegression()
    model.fit(x,y)

    
    return model
