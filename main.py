# 탐색적 데이터 분석
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
df = sns.load_dataset('seaborn')

df.index
df.columns
df.values
df.shape
df.info()
print('Column, Non-Null-Count, Dtype') 
df.describe()
df.describe(include = 'object') ##문자열의 기술통계량
df.dtypes ##column 별로 데이터 타입(int64, float64, category..) 확인

df.head()
df.tail()
df.sample(5)

df.loc[(df.loc['age'] >= 10) & (df.loc['age'] < 20)]
mask_female = df['sex'] == 'female'
df.loc[mask_female, 'column']

df.unique() ## array(unique_value)
df.nunique() ## array(unique_value_count)
df.isnull().sum()
print(
  '''
  column: null-count
  column: null-count
  '''
)
df.value_counts() ## normalize = True -> 비율확인
print(
  '''
  unique_value: count;
  '''
)

df.sort_values(ascending = False, inplace = True)

df_key = df.groupby('key')
df_key.get_group('value')
df_key.sum() #df.key.mean()

df.pivot_table(
  df,
  index = 'index로 사용할 칼럼', 
  column = 'column으로 사용할 칼럼',
  values = 'value로 나타낼 칼럼',
  aggfunc = '사용할 함수'
)
df.melt(id_vars = '남길 column', var_name='melt한 요소의 이름', value_name = 'melt한 요소의 값들을 부르는 이름')

df.plot(kind = 'bar', figsize = (10, 5))
df.plot(kind = 'barh')
df.hist()
plt.tight_layout()
plt.show()
df.plot.ked() # 확률밀도함수(히스토그램)
df.transpose().plot.bar()


# 데이터 전처리
import numpy as np
from sklearn import preprocessing

df.drop('(index)column', axis = 1) #index = axis0, column = axis1
df['dates'].dt.year
dates = df['column'].astype('float64')
dates.str.split('-').str.get(0)
df['column'].replace('-', np.nan)

for column in change_columns:
  df[column].replace(0, np.nan)
  df[column] = df[column].fillna(df[column].mean())

df_scaled = preprocessing.scale(df) ## 데이터 표준화 => array
df_scaled = pd.DataFrame(df_scaled, columns = ['column...'])

# 머신 러닝
from keras.model import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if Category:
  df = pd.get_dummies(df) # 원핫인코딩 범주형 => 0, 1
X = df.iloc[:, 0:13]
Y = df.iloc[:, [13]] #[list] => 시리즈가 아닌 데이터 프레임으로

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

model = Sequential()
model.add(Dense(30, input_dim = 13, activation = 'relu'))
##Dense(node[hidden layer], 독립변수개수, 활성화함수)
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = 'accuracy')

model.fit(X_train, Y_train, epochs = 100, verbose = 0) #epoch 실행횟수, verbose 실행 화면 보임여부

model.evaluate(X_train, Y_train)
model.evaluate(X_test, Y_test) ## train 데이터와 정확성 비교

model.predict(X.iloc[:10])
Y.iloc[:10]
## predict(새로운 학습 대상 데이터)

Y_test_predict = model.predict(X_test).round().astype('int')
confusion_matrix(Y_test, Y_test_predict)
