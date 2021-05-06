import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor

leukocytes_11=pd.read_csv('Leukocytes.csv')
WholeBlood_12=pd.read_csv('Whole blood.csv')
bone_13=pd.read_csv('Bone.csv')
buccalSwab_18=pd.read_csv('Buccal swab.csv')

leukocytes_11.set_index('Unnamed: 0',inplace=True)
WholeBlood_12.set_index('Unnamed: 0',inplace=True)
bone_13.set_index('Unnamed: 0',inplace=True)
buccalSwab_18.set_index('Unnamed: 0',inplace=True)

tissue_dfs=[leukocytes_11,WholeBlood_12,bone_13,buccalSwab_18]
from sklearn.ensemble import RandomForestRegressor

#removing source name col as the models are tissue specific
for df in tissue_dfs:
    if 'Source name' in df.columns:
        df.drop('Source name',axis=1,inplace=True)   

# ### Feature importance

import pandas as pd
leu_feat=pd.read_csv('Top400_leu_feature.csv')
leu_feat.set_index(leu_feat.iloc[:,0],inplace=True)
leu_feat.drop('Unnamed: 0',axis=1,inplace=True)

blood_feat=pd.read_csv('Top340_blood_feature.csv')
blood_feat.set_index(blood_feat.iloc[:,0],inplace=True)
blood_feat.drop('Unnamed: 0',axis=1,inplace=True)

bone_feat=pd.read_csv('Top29_bone_feature.csv')
bone_feat.set_index(bone_feat.iloc[:,0],inplace=True)
bone_feat.drop('Unnamed: 0',axis=1,inplace=True)

bs_feat=pd.read_csv('Top70_bs_feature.csv')
bs_feat.set_index(bs_feat.iloc[:,0],inplace=True)
bs_feat.drop('Unnamed: 0',axis=1,inplace=True)

new_leu_X=leukocytes_11[leu_feat.index]
new_blood_X=WholeBlood_12[blood_feat.index]
new_bone_X=bone_13[bone_feat.index]
new_bs_X=buccalSwab_18[bs_feat.index]


# ### Training after Hyperparameter tuning
from sklearn.ensemble import GradientBoostingRegressor
#leukocytes
x_train1,x_test1,y_train1,y_test1=train_test_split(new_leu_X,leukocytes_11['Age'],test_size=0.25)
gbr_leu = GradientBoostingRegressor()
gbr_leu.fit(x_train1,y_train1)
y_pred_leu=gbr_leu.predict(x_test1)
predict_train_leu=gbr_leu.predict(x_train1)

#blood
x_train2,x_test2,y_train2,y_test2=train_test_split(new_blood_X,WholeBlood_12['Age'],test_size=0.25)
rfr_blood =  TransformedTargetRegressor(regressor=RandomForestRegressor(),
                                        transformer=QuantileTransformer(),
                                        )
rfr_blood.fit(x_train2,y_train2)
y_pred_blood=rfr_blood.predict(x_test2)
predict_train_wb=rfr_blood.predict(x_train2)

#bone
x_train3,x_test3,y_train3,y_test3=train_test_split(new_bone_X,bone_13['Age'],test_size=0.25)
rfr_bone = TransformedTargetRegressor(regressor= RandomForestRegressor(),
                                        transformer=QuantileTransformer(),
                                        )
rfr_bone.fit(x_train3,y_train3)
y_pred_bone=rfr_bone.predict(x_test3)
predict_train_bone=rfr_bone.predict(x_train3)

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

#buccal_swab
x_train4,x_test4,y_train4,y_test4=train_test_split(new_bs_X,buccalSwab_18['Age'],test_size=0.25)
rfr_bclSwb = RandomForestRegressor(n_estimators= 1000, min_samples_split= 2, min_samples_leaf=1, max_features= 'sqrt', max_depth= 25)
rfr_bclSwb.fit(x_train4,y_train4)
y_pred_bclSwb=rfr_bclSwb.predict(x_test4)
predict_train_bs=rfr_bclSwb.predict(x_train4)

import pickle
pickle.dump(rfr_bclSwb,open('rfr_bclSwb.pkl','wb'))
pickle.dump(rfr_bone,open('rfr_bone.pkl','wb'))
pickle.dump(rfr_blood,open('rfr_model.pkl','wb'))
pickle.dump(gbr_leu,open('gbr_model.pkl','wb'))
