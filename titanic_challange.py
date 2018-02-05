import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def random_number_around_avg(feature: str, df: DataFrame):
    """It is used to replace the unknown values"""
    average = df[feature].mean()
    standard_deviation = df[feature].std()
    cnt_nan = df[feature].isnull().sum()

    random_ages = np.random.randint(average - standard_deviation, average + standard_deviation, size=cnt_nan)
    df.loc[np.isnan(df[feature]), feature] = random_ages


def labeling_encoding(values):
    """Replace genders by flags"""
    labelencoder_x_gender = LabelEncoder()
    values[:, 1] = labelencoder_x_gender.fit_transform(values[:, 1])

    """Tickets has 3 different class, those moved into separate columns"""
    onehotencoder = OneHotEncoder(categorical_features=[0])
    values = onehotencoder.fit_transform(values).toarray()
    values = values[:, 1:]

    """Feature scaling"""
    sc = StandardScaler()
    values = sc.fit_transform(values)
    return values


def get_x_y(df: DataFrame) -> (int, int):
    """Outputs"""
    y = None

    """Remove unnecessary features"""
    if 'Survived' in list(df.columns.values):
        y = df.iloc[:, 1].values
        df = df.drop('Survived', axis=1)
    df = df.drop('Name', axis=1)
    df = df.drop('Cabin', axis=1)
    df = df.drop('Embarked', axis=1)
    df = df.drop('Ticket', axis=1)
    df = df.drop('SibSp', axis=1)
    x = df.iloc[:, 1:].values
    x = labeling_encoding(x)
    return x, y


if __name__ == "__main__":
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    """Replace NANs"""
    random_number_around_avg("Age", train_data)
    random_number_around_avg("Age", test_data)
    random_number_around_avg("Fare", test_data)

    x_train, y_train = get_x_y(train_data)
    x_test, y_test = get_x_y(test_data)

    #x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    """Building the model"""
    # Sequential model is a linear stack of layers

    classifier = Sequential()

    # This is a common answer, but experiments can be done to find a better node number
    # hidden layer nodes number = avg(number of nodes input and output layers)
    # (11 + 1)/ (1+1) = 6
    # adding first hidden layer
    classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu', input_dim=6))

    # adding second hidden layer
    classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))

    # adding the output layer
    # softmax function if you have more than 2 categories
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # Fitting the ANN to the training set
    classifier.fit(x_train, y_train, batch_size=5, epochs=500)

    y_test_pred = classifier.predict(x_test)
    y_test_pred = (y_test_pred > 0.5)

#    cm = confusion_matrix(y_test, y_test_pred)
#    print(cm)

    y_test_pred = y_test_pred.astype(int)
    output = DataFrame(data=y_test_pred, index=test_data.iloc[:, 0], columns=['Survived'])
    output.to_csv("titanic_result2.csv")
