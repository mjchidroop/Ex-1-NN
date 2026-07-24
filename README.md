<H3>Developed by: CHIDROOP M J</H3>
<H3>Registered number: 212225240029</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 24/ 07/ 2026 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.

## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data

data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```

## OUTPUT:
### Dataset:
<img width="1287" height="437" alt="image" src="https://github.com/user-attachments/assets/58a01165-41e5-4f37-9753-351a8baee0dc" />

### Displaying first five rows:
<img width="1258" height="222" alt="image" src="https://github.com/user-attachments/assets/95a05c20-623d-451a-ad60-070138fefeb8" />

### Separating input features and output variables:
<img width="1562" height="375" alt="image" src="https://github.com/user-attachments/assets/7f769824-e463-4f2d-a75f-ecd84f9ec74d" />

### Checking null values:
<img width="1442" height="427" alt="image" src="https://github.com/user-attachments/assets/7e5888f4-2636-4f29-bf77-cab52e06f61d" />

### Checking duplicated values:
<img width="1442" height="367" alt="image" src="https://github.com/user-attachments/assets/0c8ebc99-cc12-43db-856a-9d64f0108332" />

### Displaying Statistical Information:
<img width="1427" height="468" alt="image" src="https://github.com/user-attachments/assets/9b902e00-fb1f-4c41-9a72-a9f064a6c57b" />

### Removing Categorical Columns:
<img width="1447" height="392" alt="image" src="https://github.com/user-attachments/assets/3153c821-eac6-496b-a31b-dd901951777a" />

### Normalizing the Dataset Using Min-Max Scaler:
<img width="1535" height="672" alt="image" src="https://github.com/user-attachments/assets/ce1c1308-5e18-4d51-ac8f-4549eb06a101" />

### Displaying Training Data and Testing Data:
<img width="1428" height="560" alt="image" src="https://github.com/user-attachments/assets/fb7536a2-4607-402c-8910-fea361aef63b" />

### Displaying the Length of Testing Data:
<img width="1443" height="157" alt="image" src="https://github.com/user-attachments/assets/e4b5f714-e699-43bf-9188-f2e17ae30855" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
