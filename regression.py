# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:48:06 2016

@author: GILLES Armand
"""


import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import scale, Imputer, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.stem

from utils import get_metric, get_roc_curve_cv

stem = nltk.stem.snowball.FrenchStemmer(ignore_stopwords=True)

stop = [word for word in stopwords.words('french')]

# To map niveau d'étude :
map_niveau = {"Bac +3" : 3,
                "Bac +2" : 2,
                "Bac +4" : 4,
                "Bac +1" : 1,
                "Bac +5" : 5,
                "Bac +6" : 6,
                "1er emploi" : 0,
                "Bac +7" : 7,
                u"Bac +8 et au-del\xe0" : 8,
                "Terminale" : 0}


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
    # Some outlier detection
    data.loc[data['age_p'] >= (data['age_p'].mean() + 3 * data['age_p'].std()), 'age_p'] = np.nan
    data.loc[data['age_p'] <= (data['age_p'].mean() - 3 * data['age_p'].std()), 'age_p'] = np.nan
    # col age_p have some NaN
    imp = Imputer(strategy='median', axis=1)
    data["age_p"] = pd.Series(imp.fit_transform(data["age_p"])[0])
    
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
    
    ## Analyse similitude de niveau d'étude (différence)group_formation
    # Filleul
    data["Niveau_num"] = data["Niveau"].apply(lambda x: map_niveau.get(x))
    # Parrain
    data[u"Niveau_diplome_num"] = data[u"Niveau diplôme"].apply(lambda x: map_niveau.get(x, -1)) # if Nan -> -1
    
    # Comparaison niveau Parrain / filleul
    data["niveau_etude_egal"] = data.apply(lambda row: 1 if row[u"Niveau diplôme"] == row[u"Niveau"] else 0, axis=1)
    features.append("niveau_etude_egal")
    
    # Différence de niveau Parrain / filluel    
    data["diff_niveau_etude_num"] = data["Niveau_num"] - data[u"Niveau_diplome_num"]
    features.append("diff_niveau_etude_num")
    continus_features.append("diff_niveau_etude_num")
    
    # Distance CP Parains / Filleuls
    data['distance'] = data.apply(lambda row: distance(row.lat_p, row.lon_p, row.lat_f, row.lon_f), axis=1)
    features.append("distance")
    continus_features.append("distance")
    
    
    ############## CLUSTER #################
    
    ## Clustering des filleuls
    #
    features_f = ["Sexe", "group_formation_f", "Niveau_num", "Code formation_f" ,
                  "age_f", u"Nombre de frères et soeurs"]                  
    features_f_continus = ["Niveau_num", "age_f", u"Nombre de frères et soeurs"]
    
    filleul = data[features_f].copy()
    
    # If no value then "Nombre de frères et soeurs" = 0
    filleul.loc[pd.isnull(filleul[u"Nombre de frères et soeurs"]), u"Nombre de frères et soeurs"] = 0
    for col_continus in features_f_continus:
        scaler = StandardScaler()
        filleul[col_continus] = scaler.fit_transform(filleul[col_continus].values)
        
    filleul_dummy = pd.get_dummies(filleul)
    filleul_cluster = get_kmean_cluster(filleul_dummy.values, 5, 9)
    
    data['cluster_filleul'] = filleul_cluster
    data['cluster_filleul'] = data['cluster_filleul'].astype('str')
    features.append('cluster_filleul')
    
    ## Clustering des parrains
    #
    features_p = ["H/F", "group_formation_p", "Niveau_diplome_num", "Code formation_p" ,
              "age_p", u"Secteur d'activité", "Fonction actuelle", u"A d\xe9j\xe0\xa0eu un parrainage annul\xe9 ?",
             "Temporairement indisponible"]
            # Niveau autre formation
    features_p_continus = ["Niveau_diplome_num", "age_p"]
    
    parrain = data[features_p].copy()
    
    for col_continus in features_p_continus:
        scaler = StandardScaler()
        parrain[col_continus] = scaler.fit_transform(parrain[col_continus].values)
    
    parrain_dummy = pd.get_dummies(parrain)
    parrain_cluster = get_kmean_cluster(parrain_dummy.values, 4, 0.95)
    data['cluster_parrain'] = parrain_cluster
    data['cluster_parrain'] = data['cluster_parrain'].astype('str')
    features.append('cluster_parrain')
    
        
    ####
    # others features
    features.append("Niveau")  # Filleul Bac +1, Bac + 3 etc
#    features.append(u"Niveau diplôme") # Parrain  Bac +1, Bac + 3                 # No improve
#    features.append("Mention au bac") # Filleul Bien, Trèes bien, Passable ...   # No improve
    
    
    # Drop row with nan value
    features.append('target')
    data = data[features].dropna()
    
    # Target
    y = data.target #Binary
    #y = data["Evaluation parrainage"] #Multiclass
    features.remove('target')
    
    # To set manually features
#    print "Manually set features"
#    features = ['age_f', 'cluster_filleul', 'code_formation_egal', 'formation_egal', 'projet_f_activite_p_egal']
#    continus_features = ['age_f']
    
    return data[features], features, continus_features, y
    

def distance(lat1, lon1, lat2, lon2):
#    lat1, lon1 = origin
#    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d
    
def get_kmean_cluster(X, n_cluster, n_pca):
    """
    Try to clustering with PCA / Kmean
    X : value to cluster
    n_cluster : number of cluster you want
    n_pca : Percent or number of pca you want
    Return the list of cluster
    """
    pca = PCA(n_components=n_pca).fit(X)
    X_pca = pca.transform(X)
    k_means = KMeans(init='k-means++', n_clusters=n_cluster).fit(X_pca)
    return k_means.labels_

            

def group_formation(x):
    """
    To group formation
    """
    try:
        if "ecole" in x.lower():
            return "ecole"
        elif "universit" in x.lower():
            return "universite"
        elif "institut" in x.lower():
            return "institut"
        else:
            return "autre"
    except:
        print x
        
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

print "Read city file..."
ville = pd.read_csv('data/correspondance-code-insee-code-postal.csv', sep=";")
ville = ville[["Code Postal", "geo_point_2d"]]

ville['lat'] = ville.geo_point_2d.apply(lambda x: float(x.split(",")[0]))
ville['lon'] = ville.geo_point_2d.apply(lambda x: float(x.split(", ")[1]))

ville = ville[["Code Postal", 'lat', 'lon']]

# Some citys have multiple Code psotal 
fix_cp_list = []
for idx, row in ville.iterrows():
    if "/" in row['Code Postal']:
        for cp in row['Code Postal'].split("/"):
            fix_cp_list.append({'Code Postal' : cp,
                                'lat' : row['lat'],
                                'lon' : row['lon']})
fix_cp_df = pd.DataFrame(fix_cp_list)
print "Adding %s rows to city dataset" %(str(len(fix_cp_df)))
frames = [ville, fix_cp_df]
ville = pd.concat(frames)

# Manuel fix
fix_manual_city = []
fix_manual_city.append({'Code Postal' : '75116', # 16 arrondissement Paris (2 CP)
                        'lat' : 48.860399,
                        'lon' : 2.2621})
fix_manual_city_df = pd.DataFrame(fix_manual_city)                        
                        
frames2 = [ville, fix_manual_city_df]
ville = pd.concat(frames2)

# Drop duplicated row on Code Postal
ville = ville.drop_duplicates(subset=['Code Postal'])

# Analyse cp with no match
cp_list = data["Code postal"].unique().tolist()
cp_list.extend(data["Code postal actuel"].unique().tolist())
cp_list = set(cp_list)
error_cp = [cp for cp in cp_list if cp not in ville["Code Postal"].unique().tolist()]

print "Merging with city file..."
# Parrain geo
data = pd.merge(data, ville, how='left', left_on=['Code postal actuel'], right_on=['Code Postal'])
# Filleul geo
data = pd.merge(data, ville, how='left', left_on=['Code postal'], right_on=['Code Postal'], suffixes=('_p', '_f'))

print "Process data..."
my_data, features, continus_features, y = process_data(data)


#print "Imput missing value..."
## col age_p have some NaN
#imp = Imputer(strategy='median', axis=1)
#my_data["age_p"] = pd.Series(imp.fit_transform(my_data["age_p"])[0])
#
## TO DO change outlier value from age_p
#imp = Imputer(strategy='median', axis=1)
#my_data["distance"] = pd.Series(imp.fit_transform(my_data["distance"])[0])


print "Scaling..."
for continus_col in continus_features:
        my_data[continus_col] = scale(my_data[continus_col])
        
data_save = my_data.copy()        
my_data = pd.get_dummies(my_data)

print "Spliting Dataset..."
skf = StratifiedShuffleSplit(y, 2, test_size=0.25, random_state=17)

for train_index, test_index in skf:
    X_train, X_test = my_data.iloc[train_index], my_data.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



model = LogisticRegression(class_weight='balanced', C=10, solver='lbfgs')
#model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

get_metric(y_test, y_pred, plot=False)

get_roc_curve_cv(model, my_data, y)

result = pd.DataFrame(y_test)
result.columns = ["y_test"]
result["y_pred"] = y_pred



# To fit model with all feature one by one and check CV score :
from sklearn.cross_validation import cross_val_score
score_features = []
for feature in features:
    temp_data = data_save[feature].copy()
    temp_data = pd.get_dummies(temp_data)
    
    score = cross_val_score(model, temp_data, y=y, scoring='roc_auc', cv=5)
    score_features.append({'col' : feature,
                        'score': score.mean()})
                        
df_features = pd.DataFrame(score_features)    
df_features.sort_values('score', ascending=0, inplace=True)                
#sns.barplot(x='col', y='score', data=df_features)
#plt.xticks(range(len(df_features.col)), df_features.col, rotation=75)






