import pydot as pydot
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

def read_data(path):
    data = pd.read_csv(path)
    data['Sex'] = pd.get_dummies(data['Sex'])
    data['Embarked'] = pd.get_dummies(data['Embarked'])
    return data

def run_dt(x_train, y_train, x_test, features):
    dt = DecisionTreeClassifier()
    dt.fit(x_train.values, y_train.values)
    
    export_graphviz(dt, out_file="dt.dot", class_names=['No', 'Yes'], feature_names=features, impurity=False, filled=True)
    (graph, ) = pydot.graph_from_dot_file('dt.dot', encoding='utf8')
    graph.write_png('dt.png')

    dt_prediction = dt.predict(x_test) 
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": dt_prediction})
    submission.to_csv('submission_rf.csv', index=False)

if __name__=='__main__':
    train = read_data("./titanic/train.csv")
    test = read_data("./titanic/test.csv")
    
    target_col = ['Pclass', 'Sex', 'Embarked']
    x_train = train[target_col]
    y_train = train['Survived']
    x_test = test[target_col]
    run_dt(x_train, y_train, x_test, target_col)
