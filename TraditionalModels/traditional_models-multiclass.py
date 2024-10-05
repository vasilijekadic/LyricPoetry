from lemmatizer import lemmatize
from stemmer import stem
import reldi_tokeniser
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
import seaborn as sb


def tokenize(text):
    return reldi_tokeniser.run(text, 'sr', bert=True).split()


df = pd.read_csv("LyricPoetry-multiclass.txt")
df = df.sample(frac=1).reset_index(drop=True)

corpus = df["text"]
corpus_stem = corpus.copy()
corpus_lemma = corpus.copy()
corpus_stem = corpus_stem.apply(lambda row: stem(row))
corpus_lemma = corpus_lemma.apply(lambda row: lemmatize(row))
y = df["labels"]

# MULTINOMIAL NAIVE BAYES
clf_mnb = MultinomialNB()
vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5, use_idf=False)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_df=0.9, min_df=5)
pipeline_clf = Pipeline([('vectorizer', vectorizer), ('classifier', clf_mnb)])
grid = {'classifier__alpha': [0.1, 1.0, 10, 100]}
clf = GridSearchCV(estimator=pipeline_clf, param_grid=grid, cv=StratifiedKFold(n_splits=5, shuffle=True),
                   scoring='f1_macro')

cv_results = cross_validate(clf, corpus, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(clf, corpus_stem, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(clf, corpus_lemma, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
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

cv_results = cross_validate(ovr_clf, corpus, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_stem, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_lemma, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())


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

cv_results = cross_validate(ovr_clf, corpus, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_stem, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

cv_results = cross_validate(ovr_clf, corpus_lemma, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring='f1_macro', return_train_score=True, return_estimator=True)

print("Train f1_macro: ", cv_results["train_score"].mean())
print()
print("Test f1_macro: ", cv_results["test_score"].mean())

kf = StratifiedKFold(n_splits=10, shuffle=True)
confusion_matrices_list = []
class_names = ['Ljubavna poezija', 'Misaono - refleksivna poezija', 'Rodoljubiva poezija']

for train, test in kf.split(corpus_lemma, y):
    X_train, X_test = corpus_lemma.iloc[train], corpus_lemma.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    ovr_clf.fit(X_train, y_train)
    y_pred = ovr_clf.predict(X_test)
    fold_confusion_matrices = confusion_matrix(y_test, y_pred, labels=class_names)
    confusion_matrices_list.append(fold_confusion_matrices)

confusion_matrices_array = np.array(confusion_matrices_list)
normalized_confusion_matrices = np.zeros_like(confusion_matrices_array, dtype=float)
average_confusion_matrix = np.mean(confusion_matrices_array, axis=0)
normalized_average_confusion_matrix = average_confusion_matrix.astype(float) / average_confusion_matrix.sum(axis=1, keepdims=True)

for i in range(confusion_matrices_array.shape[0]):
    normalized_confusion_matrices[i] = confusion_matrices_array[i].astype(float) / confusion_matrices_array[i].sum(axis=1, keepdims=True)

confusion_matrix_variance = np.var(normalized_confusion_matrices, axis=0)

plt.figure(figsize=(8, 6))
ax = sb.heatmap(normalized_average_confusion_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True,
            xticklabels=['Ljubavna', 'Misaono-refleksivna', 'Rodoljubiva'], yticklabels=['Ljubavna', 'Misaono-refleksivna', 'Rodoljubiva'])

for j in range(normalized_average_confusion_matrix.shape[0]):
    for k in range(normalized_average_confusion_matrix.shape[1]):
        if j == k:
            cell_color = ax.collections[0].get_cmap()(ax.collections[0].get_array().reshape(normalized_average_confusion_matrix.shape)[j, k])
            brightness = np.mean(cell_color[:3])
            text_color = 'white' if brightness < 0.6 else 'black'
            ax.text(k + 0.95, j + 0.05, f"$ \sigma $ = {np.sqrt(confusion_matrix_variance[j][k]):.2g}",
                    horizontalalignment='right', verticalalignment='top', color=text_color)
            
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
