# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:46:33 2016

@author: babou
"""

import pandas as pd
import numpy as np

# Filleul file
################

filleuls = pd.read_excel('data/Liste des filleuls VF.xlsx', encoding='utf-8')
filleuls.rename(columns={'Identifiant parrainage ':'Identifiant parrainage'}, inplace=True)


# Drop rows with no Identifiant parrainage
filleuls = filleuls.dropna(subset=["Identifiant parrainage"])

# change type to merge
filleuls["Identifiant parrainage"] = filleuls["Identifiant parrainage"].astype('int')


# Create target : 
map_eval_fil = {u"A : Le parrainage se passe tr\xe8s bien" : 1,
                'B : Le parrainage se passe bien' : 1,
                'C : Le parrainage fonctionne correctement' : 1,
                'D : Le parrainage ne se passe pas bien' : 0,
                'E : Le parrainage ne fonctionne pas' : 0}

filleuls['target'] = filleuls[u"Evaluation parrainage"].map(map_eval_fil)

# Drop if no feedback on parrains
filleuls = filleuls.dropna(subset=['target'])



# Parrain file
################

parrains = pd.read_excel('data/Liste des parrains VF.xlsx', encoding='utf-8')
parrains.rename(columns={'Identifiant parrainage ':'Identifiant parrainage'}, inplace=True)

# Drop row with no id
parrains = parrains.dropna(subset=["Identifiant parrain ", "Identifiant filleul", "Identifiant parrainage"]) # 1 row

# change type to merge
parrains["Identifiant parrainage"] = parrains["Identifiant parrainage"].astype('int')


# Merging 
###############
data = pd.merge(filleuls, parrains, how='inner', on="Identifiant parrainage", suffixes=('_f', '_p'))

data.to_csv('data/data.csv', encoding='utf-8')


