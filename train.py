"""Reproducible StudySmart AI experiment. Run: python train.py --data data/student-por.csv"""
import argparse, json
from pathlib import Path
import joblib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = ['studytime','absences','freetime','goout','activities','schoolsup','famsup','internet','higher','health']
HISTORY = BASE + ['G1','G2']

def build_pipeline(columns, model):
    categorical = [x for x in columns if x in ['activities','schoolsup','famsup','internet','higher']]
    numeric = [x for x in columns if x not in categorical]
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),('scale', StandardScaler())]), numeric),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), categorical)
    ])
    return Pipeline([('preprocess',pre),('model',model)])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/student-por.csv')
    args=parser.parse_args()
    p=Path(args.data)
    if not p.exists():
        raise SystemExit(f'Dataset missing: {p}. Download UCI Student Performance and extract student-por.csv to data/.')
    df=pd.read_csv(p,sep=';')
    required=set(HISTORY+['G3'])
    if not required.issubset(df.columns):
        raise SystemExit(f'Missing columns: {sorted(required-set(df.columns))}')
    df=df.dropna(subset=['G3'])
    if len(df)<100: raise SystemExit('Dataset too small for planned evaluation.')
    y=df['G3']
    train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=0.2,random_state=42)
    out=Path('results');out.mkdir(exist_ok=True)
    rows=[]; winner=None;best=float('inf')
    for scenario,columns in [('habits',BASE),('history',HISTORY)]:
        X=df[columns]
        models={
            'mean_baseline':DummyRegressor(strategy='mean'),
            'ridge':Ridge(alpha=10),
            'random_forest':RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=42,n_jobs=-1),
            'gradient_boosting':HistGradientBoostingRegressor(max_iter=150,l2_regularization=5,random_state=42)
        }
        cv=KFold(n_splits=5,shuffle=True,random_state=42)
        scores={}
        for name,model in models.items():
            pipe=build_pipeline(columns,model)
            s=cross_validate(pipe,X.iloc[train_idx],y.iloc[train_idx],cv=cv,scoring='neg_mean_absolute_error',n_jobs=1)
            scores[name]=float(-s['test_score'].mean())
        chosen=min(scores,key=scores.get)
        pipe=build_pipeline(columns,models[chosen]);pipe.fit(X.iloc[train_idx],y.iloc[train_idx])
        pred=np.clip(pipe.predict(X.iloc[test_idx]),0,20)
        metrics={'scenario':scenario,'selected_model':chosen,'cv_mae':scores[chosen],
                 'test_mae':float(mean_absolute_error(y.iloc[test_idx],pred)),
                 'test_rmse':float(np.sqrt(mean_squared_error(y.iloc[test_idx],pred))),
                 'test_r2':float(r2_score(y.iloc[test_idx],pred)),
                 'train_count':len(train_idx),'test_count':len(test_idx),'cv_all':scores}
        rows.append(metrics)
        joblib.dump({'pipeline':pipe,'columns':columns,'metrics':metrics},out/f'{scenario}_model.joblib')
        plt.figure(figsize=(6,5));plt.scatter(y.iloc[test_idx],pred,alpha=.55,s=22);plt.plot([0,20],[0,20],'--',color='gray');plt.xlim(0,20);plt.ylim(0,20);plt.xlabel('Actual final grade (G3)');plt.ylabel('Predicted grade');plt.title(f'{scenario.title()} model: held-out test set');plt.tight_layout();plt.savefig(out/f'{scenario}_predictions.png',dpi=170);plt.close()
        imp=permutation_importance(pipe,X.iloc[test_idx],y.iloc[test_idx],scoring='neg_mean_absolute_error',n_repeats=10,random_state=42)
        importance=pd.Series(imp.importances_mean,index=columns).sort_values()
        plt.figure(figsize=(7,5));importance.plot.barh();plt.xlabel('Increase in MAE when shuffled (grade points)');plt.title(f'{scenario.title()} model: permutation importance');plt.tight_layout();plt.savefig(out/f'{scenario}_importance.png',dpi=170);plt.close()
        if metrics['test_mae']<best: best=metrics['test_mae'];winner=scenario
    with (out/'metrics.json').open('w') as f:json.dump({'dataset_rows':len(df),'selection_method':'5-fold CV on training only','split_seed':42,'test_fraction':.2,'scenarios':rows,'best_test_scenario_descriptive_only':winner},f,indent=2)
    pd.DataFrame([{k:v for k,v in r.items() if k!='cv_all'} for r in rows]).to_csv(out/'summary.csv',index=False)
    print(pd.read_csv(out/'summary.csv').to_string(index=False))
    print('Saved models, plots and metrics in results/')

if __name__=='__main__':main()
