from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



def train_decision_tree_model(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    return accuracy
