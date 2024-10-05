from lemmatizer import lemmatize
from stemmer import stem
import reldi_tokeniser
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
import seaborn as sb


def tokenize(text):
    return reldi_tokeniser.run(text, 'sr', bert=True).split()


df = pd.read_csv("dataset-multilabel.txt")
mlb = MultiLabelBinarizer()
mlb_result = mlb.fit_transform([str(df.loc[i, 'labels']).split(',') for i in range(len(df))])
df = pd.concat([df['text'], pd.DataFrame(mlb_result, columns=list(mlb.classes_))], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

corpus = df["text"].copy()
corpus_stem = corpus.copy()
corpus_lemma = corpus.copy()
corpus_stem = corpus_stem.apply(lambda row: stem(row))
corpus_lemma = corpus_lemma.apply(lambda row: lemmatize(row))
y = df.loc[:, "Ljubavna poezija":"Rodoljubiva poezija"]

# MULTINOMIAL NAIVE BAYES
binary_clf = MultinomialNB()
vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5, use_idf=False)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5)
pipeline_clf = Pipeline([('vectorizer', vectorizer), ('classifier', binary_clf)])
grid = {'classifier__alpha': [0.1, 1.0, 10, 100]}
clf = GridSearchCV(estimator=pipeline_clf, param_grid=grid, cv=StratifiedKFold(n_splits=5, shuffle=True),
                   scoring='f1')
ovr_clf = OneVsRestClassifier(clf)

cv_results = cross_validate(ovr_clf, corpus, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_stem, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_lemma, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())


# LOGISTIC REGRESSION
binary_clf = LogisticRegression(max_iter=10000, class_weight='balanced')
vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5, use_idf=False)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5)

pipeline_clf = Pipeline([('vectorizer', vectorizer), ('svd', TruncatedSVD(n_components=100, random_state=42)),
                         ('classifier', binary_clf)])

grid = {'classifier__C': [0.1, 1.0, 10, 100]}

clf = GridSearchCV(estimator=pipeline_clf, param_grid=grid, cv=StratifiedKFold(n_splits=5, shuffle=True),
                   scoring='f1')
ovr_clf = OneVsRestClassifier(clf)

cv_results = cross_validate(ovr_clf, corpus, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_stem, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_lemma, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

kf = IterativeStratification(n_splits=10, order=1)
confusion_matrices_list = []

for train, test in kf.split(corpus_lemma, y):
    X_train, X_test = corpus_lemma.iloc[train], corpus_lemma.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    ovr_clf.fit(X_train, y_train)
    y_pred = ovr_clf.predict(X_test)
    fold_confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)
    confusion_matrices_list.append(fold_confusion_matrices)

confusion_matrices_array = np.array(confusion_matrices_list)
normalized_confusion_matrices = np.zeros_like(confusion_matrices_array, dtype=float)
average_confusion_matrices = np.mean(confusion_matrices_array, axis=0)
normalized_average_confusion_matrices = np.zeros_like(average_confusion_matrices)

for i in range(average_confusion_matrices.shape[0]):
    row_sums = average_confusion_matrices[i].sum(axis=1, keepdims=True)
    normalized_average_confusion_matrices[i] = average_confusion_matrices[i] / row_sums

for i in range(confusion_matrices_array.shape[0]):
    for j in range(confusion_matrices_array.shape[1]):
        row_sums = confusion_matrices_array[i][j].sum(axis=1, keepdims=True)
        normalized_confusion_matrices[i][j] = confusion_matrices_array[i][j] / row_sums

confusion_matrices_variance = np.var(normalized_confusion_matrices, axis=0)

class_names = ['Ljubavna poezija', 'Misaono - refleksivna poezija', 'Rodoljubiva poezija']
fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))
cms = normalized_average_confusion_matrices

for i, (cm, ax) in enumerate(zip(cms, axes)):
    sb.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax)

    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            if j == k:
                cell_color = ax.collections[0].get_cmap()(ax.collections[0].get_array().reshape(cm.shape)[j, k])
                brightness = np.mean(cell_color[:3])
                text_color = 'white' if brightness < 0.6 else 'black'
                ax.text(k + 0.95, j + 0.05, f"$ \sigma $ = {np.sqrt(confusion_matrices_variance[i][j][k]):.2g}",
                        horizontalalignment='right', verticalalignment='top', color=text_color)

    ax.set_title(f'{class_names[i]}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.show()



# SUPPORT VECTOR MACHINE
binary_clf = LinearSVC(max_iter=10000, class_weight='balanced')
vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5, use_idf=False)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5)

pipeline_clf = Pipeline([('vectorizer', vectorizer), ('svd', TruncatedSVD(n_components=100, random_state=42)),
                         ('classifier', binary_clf)])

grid = {'classifier__C': [0.1, 1.0, 10, 100]}

clf = GridSearchCV(estimator=pipeline_clf, param_grid=grid, cv=StratifiedKFold(n_splits=5, shuffle=True),
                   scoring='f1')
ovr_clf = OneVsRestClassifier(clf)

cv_results = cross_validate(ovr_clf, corpus, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_stem, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_lemma, y, cv=IterativeStratification(n_splits=10, order=1),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())