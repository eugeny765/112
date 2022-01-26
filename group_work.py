#создайте здесь свой групповой проект!#создай здесь свой индивидуальный проект!
import pandas as pd 
df = pd.read_csv('train.csv')


df.drop(['id','education_status','bdate','has_photo','has_mobile','followers_count','graduation','life_main','people_main','city','occupation_name','career_start','career_end','last_seen','relation'], axis = 1, inplace = True)

def sexapply(sex):
    if sex == 2:
        return 0
    return 1
df['sex'] = df['sex'].apply(sexapply)
df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)

print(df['langs'].value_counts())
def Langsapply(langs):
    if langs.find('English') != -1:
        return 1
    else:
        return 0
df['langs'] = df['langs'].apply(Langsapply)
df['occupation_type'].fillna('university', inplace = True)
def occupation_typeapply(occupation_type):
    if occupation_type == 'university':
        return 0
    else:
        return 1
df['occupation_type'] = df['occupation_type'].apply(occupation_typeapply)
df.info()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x =  df.drop('result', axis = 1)
y = df['result']
xtrain , xtest , ytrain , ytest = train_test_split(x,y, test_size = 0.25)


sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(xtrain , ytrain)

y_pred = classifier.predict(xtest)
print('Процент правильно предсказанных исходов:', round(accuracy_score(ytest,y_pred),2)*100)
print('Confusion matrix:')
print(confusion_matrix(ytest,y_pred))