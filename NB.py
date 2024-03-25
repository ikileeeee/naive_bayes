import pandas as pd
import numpy as np
from scipy.stats import norm
import math


def fit(data, izlazna, smoothing=1e-5):
    model={}
    vrednost=data[izlazna].value_counts()
    vrednost=(vrednost+smoothing)/(vrednost.sum()+smoothing)
    model['Izlazna']=np.log(vrednost)
    
    for kolona in data.drop(izlazna, axis=1).columns:
        if data[kolona].dtype=='object':
            matr_konf=pd.crosstab(data[kolona], data[izlazna])+smoothing
            matr_konf=matr_konf/matr_konf.sum(axis=0)
            matr_konf=np.log(matr_konf)
            matr_konf[np.isneginf(matr_konf)] = 0
            model[kolona]=matr_konf
        else:
            matr_konf=data.groupby(izlazna)[kolona].agg(['mean', 'std'])
            model[kolona]=matr_konf
    return model

def predict(model, instanca):
    verovatnoce_po_klasi={}
    
    for vrednost_izlazne in model['Izlazna'].index:
        verovatnoca=0
        
        for atribut in model:
            if atribut=='Izlazna':
                verovatnoca= verovatnoca+model['Izlazna'][vrednost_izlazne]
            elif 'mean' in model[atribut] and 'std' in model[atribut]:
                exponent = -((instanca[atribut] - model[atribut]['mean'][vrednost_izlazne]) ** 2) / (2 * model[atribut]['std'][vrednost_izlazne] ** 2)
                coefficient = 1 / (math.sqrt(2 * math.pi) * model[atribut]['std'][vrednost_izlazne])
                pdf = coefficient * math.exp(exponent)
                if pdf==0:
                   pdf = 1
                verovatnoca+=np.log(pdf)
            else:
                verovatnoca= verovatnoca+model[atribut][vrednost_izlazne][instanca[atribut]]

        verovatnoce_po_klasi[vrednost_izlazne]=verovatnoca
    zakljucak=max(verovatnoce_po_klasi, key=verovatnoce_po_klasi.get)
    return zakljucak, verovatnoce_po_klasi
    
    
data=pd.read_csv('data/drug.csv')

model=fit(data, 'Drug')

data_new = [
    {'Age': 33, 'Sex': 'M', 'BP': 'LOW', 'Cholesterol': 'NORMAL', 'Na': 0.8, 'K': 0.6},
    {'Age': 45, 'Sex': 'F', 'BP': 'HIGH', 'Cholesterol': 'HIGH', 'Na': 0.512, 'K': 0.8},
    {'Age': 23, 'Sex': 'F', 'BP': 'NORMAL', 'Cholesterol': 'NORMAL', 'Na': 0.772, 'K':0.8},
]
data_new= pd.DataFrame(data_new, columns=['Age', 'Sex', 'BP', 'Cholesterol','Na', 'K'])

for i in range(len(data_new)):
	prediction, confidence = predict(model, data_new.iloc[i])

	data_new.loc[i,'prediction'] = prediction
	for klasa in confidence:
		data_new.loc[i,'class='+klasa] = confidence[klasa]   
print(data_new['prediction'])

for i in range(len(data)):
	prediction, confidence = predict(model, data.iloc[i])

	data.loc[i,'prediction'] = prediction
	for klasa in confidence:
		data.loc[i,'class='+klasa] = confidence[klasa]   
print(data['prediction'])

    