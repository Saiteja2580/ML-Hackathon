from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



def train_decision_tree_model(x,y):
    

    model = DecisionTreeClassifier(random_state=42)
    model.fit(x,y)

    
    return model

