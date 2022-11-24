
import numpy as np

from scpye.detection.pipeline_factory import (create_image_pipeline,
                                              create_feature_pipeline)
from scpye.detection.training import (create_voting_classifier, cross_validate_classifier)
from scpye.utils.data_manager import DataManager

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'

# %%

train_inds = range(2, 12, 3) + range(1, 12, 3)
test_inds = range(0, 12, 3)

k = 0.5
pmin = 28
bbox = np.array([300, 0, 600, 1440])
loc = True
cspace = ["hsv", "lab"]
method = 'lr'
patch = True

# %%
drd = DataManager(base_dir, color=color, mode=mode, side=side)
img_ppl = create_image_pipeline(bbox=bbox, k=k)
ftr_ppl = create_feature_pipeline(pmin=pmin, cspace=cspace, loc=loc,
                                  patch=patch)

Is, Ls = drd.load_image_label_list(train_inds)
Its, Lts = img_ppl.transform(Is, Ls)
Xt, yt = ftr_ppl.fit_transform(Its, Lts)

# %%
# Train a logistic regression classifier
# X_t, X_v, y_t, y_v = train_test_split(Xt, yt, test_size=0.3)
# param_grid = [{'C': [0.1, 1, 10, 100]}]
# param_grid = [{'n_estimators': [10, 30, 50]}]
# param_grid = {'lr__C': [50, 200], 'rf__n_estimators': [20, 50],
#              'svc__C': [50, 200]}
# clf1 = RandomForestClassifier()
# clf2 = LogisticRegression()
# clf3 = SVC(probability=True)
# clf4 = GaussianNB()
# eclf = VotingClassifier(estimators=[('rf', clf1),
#                                    ('lr', clf2),
#                                    ('svc', clf3)], voting='soft')
# grid = GridSearchCV(estimator=eclf, param_grid=param_grid, cv=4, verbose=5,
#                    scoring="f1_weighted")
# grid.fit(X_t, y_t)
clf, param_grid = create_voting_classifier()
grid = cross_validate_classifier(Xt, yt, clf, param_grid)

# %%
I, L = drd.load_image_and_label(3)

X = img_ppl.transform(I)
X = ftr_ppl.transform(X)
y = img_ppl.transform(L[..., 1])

mask = ftr_ppl.named_steps['remove_dark'].mask

y_pred = grid.predict(X)
bw = mask.copy()
bw[bw > 0] = y_pred

# y_proba = grid.predict_proba(X)
# proba = np.array(mask, dtype=np.float)
# proba[proba > 0] = y_proba[:, 1]

# %%
# Plot pr curve
# y_true = y.ravel()
# probas_pred = proba.ravel()
# precision, recall, thresholds = precision_recall_curve(y_true, probas_pred,
#                                                       pos_label=1)
# plt.plot(recall, precision)
# Plot roc/auc curve
