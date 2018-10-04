import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import matplotlib as plt
from sklearn.preprocessing import KBinsDiscretizer

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

"""
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
"""

y = train_data.Survived

# obvious features, Sib maybe not
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

X_init = train_data[features]
#
# X_init.hist(column='Age')
# plt.show()

# X_dtype_cat = [cname for cname in X_init.columns if X_init.dtype in ['object']]

# X_dtype_num = [cname for cname in X_init.columns if X_init.dtype in ['int64', 'float64']]

X_init = pd.get_dummies(X_init)

imputer = SimpleImputer()

X = imputer.fit_transform(X_init)
"""
things to think about 
1. categories for fare and age...how to figure out the best splits
"""
rf_model = RandomForestClassifier(random_state=0)

rf_model.fit(X, y)

preds = rf_model.predict(X)


rf_score = cross_val_score(rf_model, X, y, scoring='neg_mean_absolute_error', cv=5)
print('RF Mean Absolute Error %2f' %(-1 * rf_score.mean()))

# export test_data
Xt = test_data[features]
Xt = pd.get_dummies(Xt)
Xt_imp = pd.DataFrame(imputer.fit_transform(Xt))

# age
X_fare = Xt_imp[:1]

for i in (2, 6):
    binner = KBinsDiscretizer(n_bins=i, encode='onehot')
    X_fare_bin = binner.fit_transform(X_fare)
    Xt_imp.drop(1, axis=1)
    X = Xt_imp + X_fare_bin
    test_model(X, y)


# test_preds = xgb_model.predict(Xt_imp)
#
# submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_preds})
# submission.to_csv("submission_xgb.csv", index=False)


def test_model(x, y, n=1000, lr=0.02):
    xgb_model = XGBClassifier(n_estimators=n, learning_rate=lr)
    xgb_model.fit(X, y, verbose=False)

    xgb_score = cross_val_score(xgb_model, x, y, scoring='neg_mean_absolute_error', cv=5)
    print('XGB LR=%f, N=%d Absolute Error %2f' % (lr, n, -1 * xgb_score.mean()))
