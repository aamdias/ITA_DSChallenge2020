#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Bibliotecas Utilizadas no Pre Processamento
import pandas as pd
import numpy as np
import matplotlib.pyplot as ply
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_cell_magic('capture', '', 'from tqdm import tqdm_notebook as tqdm\ntqdm().pandas()')


# In[3]:


# Importanto dados utilizados
messages = pd.read_csv('databases/private-data/messages.csv')
orders = pd.read_csv('databases/private-data/order.csv')
sensors = pd.read_csv('databases/private-data/sensors.csv')

# Fetching public data
public = pd.read_csv('databases/public.csv')


# ### Reduzindo o problema para 100 voos

# In[4]:


# Escolhendo voos
public_reduced = public.head(100).copy()

# Armazenando os voos que serão previstos
flights = set(public_reduced['FLIGHT'].unique())


# In[5]:


# Reduzindo sensors e messages para somente os voos estudados
sensors_reduced = sensors[sensors['FLIGHT'].isin(flights)].copy()
messages_reduced = messages[messages['FLIGHT'].isin(flights)].copy()

# Tirando as duplicatas
sensors_reduced.drop_duplicates(inplace=True)
messages_reduced.drop_duplicates(inplace=True)


# In[6]:


# Vendo as features de sensors
sensors_reduced.describe()


# In[7]:


# Retirando as colunas nulas
drop_columns = ['WAR_SYS_1','COM_SYS_1','WAR_SYS_2','WAR_SYS_3']
sensors_reduced.drop(drop_columns,axis=1,inplace=True)

# Retirando outras colunas (baixa variância) em relação  a média
other_columns = ['PAR_AC_3','PAR_AC_4','PAR_SYS_5','PAR_SYS_6']
sensors_reduced.drop(other_columns,axis=1,inplace=True)


# In[8]:


# Escolhendo 6 features
sensors_reduced = sensors_reduced[['AC','FLIGHT','TIME','AMBIENT_1','PAR_SYS_1','PAR_SYS_9']]


# In[9]:


# Vendo as features de messages
messages_reduced.describe()


# ### Removendo outliers numéricos de cada uma das features através da análise de histogramas

# #### Tabela Sensors Reduced

# In[10]:


# TIME
sns.distplot(sensors_reduced['TIME'])


# Calda longa! Precisa remover outliers

# In[11]:


# Removendo os outliers
lowerbound,upperbound = np.percentile(sensors_reduced['TIME'],[1,99])
sensors_reduced = sensors_reduced.query('{} <= TIME <= {}'.format(lowerbound,upperbound))

# Nova Distribuição para TIME
sns.distplot(sensors_reduced['TIME'])


# In[12]:


# AMBIENT_1
sns.distplot(sensors_reduced['AMBIENT_1'])


# Não precisa remover outliers

# In[13]:


# PAR_SYS_1
sns.distplot(sensors_reduced['PAR_SYS_1'])


# Caldas longas em ambos os lados. Haverá remoção de outliers!

# In[14]:


# Removendo os outliers
lowerbound,upperbound = np.percentile(sensors_reduced['PAR_SYS_1'],[1,99])
sensors_reduced = sensors_reduced.query('{} <= PAR_SYS_1 <= {}'.format(lowerbound,upperbound))

# Nova Distribuição para PAR_SYS_1
sns.distplot(sensors_reduced['PAR_SYS_1'])


# In[15]:


# PAR_SYS_9
sns.distplot(sensors_reduced['PAR_SYS_9'])


# Não há outliers

# In[16]:


# Guardando a tabela sensors_reduced sem outliers
wo_sensors_reduced = sensors_reduced.copy()


# #### Tabela Messages Reduced

# In[17]:


# TIME
sns.distplot(messages_reduced['TIME'])


# Sem Outliers!

# In[18]:


# FLIGHT_PHASE
sns.distplot(messages_reduced['FLIGHT_PHASE'])


# Sem outliers!

# In[19]:


# MESSAGE
sns.distplot(messages_reduced['MESSAGE'])


# Sem outliers!

# In[20]:


# Guardando a tabela sensors_reduced sem outliers
wo_messages_reduced = messages_reduced.copy()


# ### Montando a tabela com as features e a target

# In[21]:


# Formatando as tabelas para merge
wo_sensors_reduced.set_index('FLIGHT',inplace=True)
wo_messages_reduced.set_index('FLIGHT',inplace=True)

# Merging sensors and messages data
sns_and_msg = wo_sensors_reduced.merge(wo_messages_reduced,left_on="FLIGHT", right_on="FLIGHT",how="inner",suffixes=('_sns','_msg')).progress_apply(lambda x: x)
sns_and_msg.drop('AC_msg',axis=1,inplace=True)

# Formatando a tabela public reduced para merge
public_reduced.set_index('FLIGHT',inplace=True)

# Acrescendo a coluna target através de um merge com a tabelas public
model_data = sns_and_msg.merge(public_reduced,left_on="FLIGHT",right_on="FLIGHT",how="inner",suffixes=('_sm','_pb')).progress_apply(lambda x: x)


# In[22]:


# Ajustando índice
model_data.reset_index(inplace=True)

# Renomeando as colunas
model_data.columns = ['flight','ac','sensor_time','sensor_ambient',
                      'sensor_par_sys_1','sensor_par_sys_9',
                      'message_time','flight_phase','message_type',
                      'message','target']


# In[23]:


model_data


# ### Encoding de Variáveis Categóricas (Target Encoding)

# In[24]:


# Filtrando somente os colunas categóricas
num_cols = model_data._get_numeric_data().columns
cat_cols = set(model_data.columns) - set(num_cols) # Armazenando colunas categóricas


# In[25]:


# Printando as variáveis categóricas
cat_cols


# In[26]:


# Função auxiliar para calcular média suavizada (método escolhido para Target Encoding)
def calc_smooth_mean(df,cat_name,target,weight):
    # Compute the global mean
    mean = df[target].mean()
    
    # Compute the number of values and the mean of each group
    agg = df.groupby(cat_name)[target].agg(['count','mean'])
    counts = agg['count']
    means = agg['mean']
    
    # Compute the smoothed means
    smooth = (counts*means + weight*mean)/(counts+weight)
    
    return df[cat_name].map(smooth)


# In[27]:


# Setando o peso substituindo as colunas categóricas pelo seu respectivo encoding
WEIGHT = 4
model_data['ac'] = calc_smooth_mean(model_data,cat_name='ac',target='target',weight=WEIGHT)
model_data['flight'] = calc_smooth_mean(model_data,cat_name='ac',target='target',weight=WEIGHT)
model_data['message_type'] = calc_smooth_mean(model_data,cat_name='ac',target='target',weight=WEIGHT)


# ### Finalmente após todo o processo de Feature Engineering

# In[28]:


model_data


# ## Implementando o modelo XGBoost

# In[29]:


# Bibliotecas para rodar o modelo e testar sua eficiência
from sklearn.model_selection import train_test_split
import xgboost as xgb


# In[30]:


# Separando as features e a target
X,y = model_data.iloc[:,:-1],model_data.iloc[:,-1]


# In[31]:


# Separando Train e Validation sets
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=41)


# In[32]:


# Instaciando o XGBoost Regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,verbosity=1,
                max_depth = 5, alpha = 10, n_estimators = 10)


# In[33]:


# Por fim, treinando o modelo
xg_reg.fit(X_train,y_train)


# In[34]:


from sklearn.metrics import mean_squared_error
# Testando a acurácia do modelo com a validation data
pred_train = xg_reg.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, pred_train))
print("RMSE: %f" % (rmse))


# ### Previsão dos dados sem Label

# In[35]:


# Separando base de dados para teste (com label desconhecido)
public_flights = set(public['FLIGHT'].unique())
sensors_flights = set(sensors['FLIGHT'].unique())

test_flights = sensors_flights - public_flights


# #### Preparando o vetor das features para esses dados

# In[36]:


# Slicing
sensors_test = sensors[sensors['FLIGHT'].isin(test_flights)].copy()
messages_test = messages[messages['FLIGHT'].isin(test_flights)].copy()


# In[37]:


# Tirando as duplicatas
sensors_test.drop_duplicates(inplace=True)
messages_test.drop_duplicates(inplace=True)


# In[38]:


# Retirando colunas nulas
drop_columns = ['WAR_SYS_1','COM_SYS_1','WAR_SYS_2','WAR_SYS_3','PAR_AC_3','PAR_AC_4','PAR_SYS_5','PAR_SYS_6']
sensors_test.drop(drop_columns,axis=1,inplace=True)

# Escolhendo somente as colunas importantes
sensors_test = sensors_test[['AC','FLIGHT','TIME','AMBIENT_1','PAR_SYS_1','PAR_SYS_9']]


# In[ ]:


# Preparando df para merge
sensors_test.set_index('FLIGHT',inplace=True)
messages_test.set_index('FLIGHT',inplace=True)

# Merging sensors and messages (Usando tqdm para mostrar progresso)
test_data = sensors_test.merge(messages_test,left_on="FLIGHT", right_on="FLIGHT",how="inner",suffixes=('_sns','_msg')).progress_apply(lambda x: x)
test_data.drop('AC_msg',axis=1,inplace=True)

# Resetando o índice
test_data.reset_index()


# In[ ]:


# Renomeando as colunas
test_data.columns = ['ac','flight','sensor_time','sensor_ambient',
                      'sensor_par_ac_1','sensor_par_ac_2','sensor_par_sys_1',
                      'sensor_par_sys_2','sensor_par_sys_3','sensor_par_sys_4',
                      'sensor_par_sys_9','sensor_par_sys_10','sensor_par_sys_7',
                      'sensor_par_sys_8','message_time','flight_phase','message_type',
                      'message']

# Filtrando somente os colunas categóricas
num_cols = test_data._get_numeric_data().columns
cat_cols = set(test_data.columns) - set(num_cols) # Armazenando colunas categóricas

# Setando o peso substituindo as colunas categóricas pelo seu respectivo encoding
WEIGHT = 4
test_data['ac'] = calc_smooth_mean(test_data,cat_name='ac',target='target',weight=WEIGHT)
test_data['flight'] = calc_smooth_mean(test_data,cat_name='ac',target='target',weight=WEIGHT)
test_data['message_type'] = calc_smooth_mean(test_data,cat_name='ac',target='target',weight=WEIGHT)


# ### Fazendo a previsão e salvando em um DF

# In[ ]:


final_prediction = xg_reg.predict(test_data)


# In[ ]:


final_prediction

