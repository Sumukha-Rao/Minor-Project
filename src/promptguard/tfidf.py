from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from .splitting import group_folds
from .utils import write_json, measured


def train_tfidf(train, config, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=config['tfidf']
    pipeline=Pipeline([('tfidf',TfidfVectorizer(lowercase=False,strip_accents=None,sublinear_tf=True)),
                       ('lr',LogisticRegression(max_iter=settings['max_iter'],solver='liblinear',random_state=config['seed']))])
    if settings['search']=='smoke':
        grid=[{'tfidf__analyzer':['word'],'tfidf__ngram_range':[(1,2)],'tfidf__min_df':[1],
               'tfidf__max_features':[5000],'lr__C':[1.0],'lr__class_weight':['balanced']}]
    else:
        # Full mode expands each axis. Compact still tests word/char, vocabulary,
        # min_df, regularization, and weighting using four planned configurations.
        grid=[{'tfidf__analyzer':['word'],'tfidf__ngram_range':[(1,2)],'tfidf__min_df':[1],
               'tfidf__max_features':[20000],'lr__C':[1.0],'lr__class_weight':[None]},
              {'tfidf__analyzer':['word'],'tfidf__ngram_range':[(1,2)],'tfidf__min_df':[2],
               'tfidf__max_features':[50000],'lr__C':[.5],'lr__class_weight':['balanced']},
              {'tfidf__analyzer':['char'],'tfidf__ngram_range':[(3,5)],'tfidf__min_df':[2],
               'tfidf__max_features':[30000],'lr__C':[1.0],'lr__class_weight':['balanced']},
              {'tfidf__analyzer':['char'],'tfidf__ngram_range':[(3,5)],'tfidf__min_df':[1],
               'tfidf__max_features':[50000],'lr__C':[4.0],'lr__class_weight':[None]}]
        if settings['search']=='full':
            for g in grid:
                g.update({'lr__C':[.1,1.,10.],'lr__class_weight':[None,'balanced'],
                          'tfidf__min_df':[1,2,5],'tfidf__max_features':[20000,50000]})
    folds=group_folds(train,settings['cv_folds'],config['seed'])
    search=GridSearchCV(pipeline,grid,cv=folds,scoring='f1_macro',n_jobs=settings['n_jobs'],refit=True,error_score='raise')
    with measured() as costs: search.fit(train.text.tolist(),train.label.to_numpy())
    joblib.dump(search.best_estimator_,output/'pipeline.joblib')
    pd.DataFrame(search.cv_results_).to_csv(output/'cv_results.csv',index=False)
    write_json(output/'training.json',{'best_parameters':search.best_params_,'cv_macro_f1':search.best_score_,**costs})
    return search.best_estimator_, costs
