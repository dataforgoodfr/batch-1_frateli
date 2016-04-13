# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:48:06 2016

@author: GILLES Armand
"""


import pandas as pd
import numpy as np

from sklearn.preprocessing import scale, Imputer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.stem

from utils import get_metric

stem = nltk.stem.snowball.FrenchStemmer(ignore_stopwords=True)

stop = [word for word in stopwords.words('french')]


def process_data(data):
    
    
    features = [] # list of features created
    continus_features = []
    
    ## Same Sexe
    data["sexe_egal"] = data.apply(lambda row: 1 if row["H/F"] == row["Sexe"] else 0, axis=1)
    features.append("sexe_egal")
    
    ## Age du Filleul
    data['age_f'] = pd.datetime.now().year - pd.to_datetime(data["Date de naissance_f"], errors='coerce').dt.year
    features.append("age_f")
    continus_features.append("age_f")
    
    ## Age du Filleul
    data['age_p'] = pd.datetime.now().year - pd.to_datetime(data["Date de naissance_p"], errors='coerce').dt.year
    
    features.append("age_p")
    continus_features.append("age_p")
    
    ## Formation Filleul
    data["group_formation_f"] = data["Formation actuelle"].apply(lambda x: group_formation(x))
    features.append("group_formation_f")
    
    ## Formation parrain
    # Rename col
    data.rename(columns={u'Dipl\xf4me Principal':'Diplome Principal'}, inplace=True)
    data["group_formation_p"] = data["Diplome Principal"].apply(lambda x: group_formation(x))
    features.append("group_formation_p")
    
    ## Similitude formation Parrain / Filleul
    data['formation_egal'] = data.apply(lambda row: 1 if row["group_formation_p"] == row["group_formation_f"] else 0, axis=1)
    features.append("formation_egal")

    ## Similitude Code Formation Parrain / Filleul    
    data["code_formation_egal"] = data.apply(lambda row: 1 if row["Code formation_f"] == row["Code formation_p"] else 0, axis=1)
    features.append("code_formation_egal")
    
    ## Analyse projet motivation filleul / activité parrain
    data["projet_f_activite_p_egal"] = data.apply(lambda row: get_similitude_projet_activite(row["Projet professionnel"],
                                                                                             row[u"Secteur d'activité"]),
                                                    axis=1)
    features.append("projet_f_activite_p_egal")
    
    ## Analyse projet motivation filleul / activité parrain précédente
    data["projet_f_activite_pre_p_egal"] = data.apply(lambda row: get_similitude_projet_activite(row["Projet professionnel"],
                                                                                                row[u"Fonction précédente"]),
                                                     axis=1)
    features.append("projet_f_activite_pre_p_egal")
    
    ## Analyse projet motivation filleul / activité parrain (précédent ou non)
    data['projet_f_activite_p_egal_all'] = 0
    data.loc[(data["projet_f_activite_p_egal"] == 1) | (data["projet_f_activite_pre_p_egal"] == 1), 'projet_f_activite_p_egal_all'] = 1
    features.append("projet_f_activite_p_egal_all")
    
    ####
    # others features
    features.append("Niveau")  # Bac +1, Bac + 3 etc
    
    return data[features], features, continus_features 
            

def group_formation(x):
    """
    To group formation
    """
    if "ecole" in x.lower():
        return "ecole"
    elif "universit" in x.lower():
        return "universite"
    elif "institut" in x.lower():
        return "institut"
    else:
        return "autre"
        
def get_similitude_projet_activite(projet, activite):
    """
    projet : motivation du filleul
    activite : activité du parrain
    -----------
    Stem all word in description.
    If match with this stem description (1 word activite in projet) return 1 else 0.
    
    If projet or activite is NaN, return 0
    """
    if projet is np.nan:
        return 0
    if activite is np.nan:
        return 0
    projet_stem = [stem.stem(word) for word in word_tokenize(projet) if word not in stop]
    activite_stem = [stem.stem(word) for word in word_tokenize(activite) if word not in stop]
    
    if len(set(activite_stem).intersection(projet_stem)) >= 1:
        return 1
    else:
        return 0    

print "Read data..."
data = pd.read_csv('data/data.csv', encoding='utf-8')

y = data.target

my_data, features, continus_features = process_data(data)


print "Imput missing value..."
# col age_p have some NaN
imp = Imputer(strategy='median', axis=1)
my_data["age_p"] = pd.Series(imp.fit_transform(my_data["age_p"])[0])

# TO DO change outlier value from age_p

print "Scaling..."
for continus_col in continus_features:
        my_data[continus_col] = scale(my_data[continus_col])
        
my_data = pd.get_dummies(my_data)

print "Spliting Dataset..."
skf = StratifiedShuffleSplit(y, 2, test_size=0.25, random_state=17)

for train_index, test_index in skf:
    X_train, X_test = my_data.iloc[train_index], my_data.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

get_metric(y_test, y_pred, plot=True)

result = pd.DataFrame(y_test)
result.columns = ["y_test"]
result["y_pred"] = y_pred









