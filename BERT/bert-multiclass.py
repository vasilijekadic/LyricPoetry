import os
import shutil
from math import ceil
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, confusion_matrix
from transformers import EvalPrediction
import seaborn as sb
from matplotlib import pyplot as plt


def preprocess_data(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    labels = examples["labels"]
    encoding["labels"] = labels
    return encoding


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    f1 = f1_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds).tolist()
    return {"confusion_matrix": cm, "f1": f1}


df = pd.read_csv("LyricPoetry-multiclass.txt")
df = df.sample(frac=1).reset_index(drop=True)

label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['labels'])

dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("classla/bcms-bertic")
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

labels = ['Ljubavna poezija', 'Misaono - refleksivna poezija', 'Rodoljubiva poezija']
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

encoded_dataset = dataset.map(preprocess_data, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

fold_num = 10
fold_inner_num = 5
k_fold = StratifiedKFold(n_splits=fold_num, shuffle=True)
corpus = df["text"].copy()
y = df["labels"].copy()

f1_mean = 0
confusion_matrices_list = []
f1_scores_average = np.zeros(10, dtype=float)

for train, test in k_fold.split(corpus, y):
    df_train = df.loc[train].copy()
    df_test = df.loc[test].copy()
    df_train = df_train.reset_index(drop=True)

    f1_scores = np.zeros(10, dtype=float)
    k_fold_inner = StratifiedKFold(n_splits=fold_inner_num, shuffle=True)
    for train_inner, test_inner in k_fold_inner.split(df_train["text"], df_train["labels"]):
        df_train_inner = df_train.loc[train_inner].copy()
        df_test_inner = df_train.loc[test_inner].copy()

        dataset_train_inner = Dataset.from_pandas(df_train_inner)
        dataset_test_inner = Dataset.from_pandas(df_test_inner)

        encoded_dataset_train_inner = dataset_train_inner.map(preprocess_data, batched=True,
                                                              remove_columns=dataset.column_names)
        encoded_dataset_test_inner = dataset_test_inner.map(preprocess_data, batched=True,
                                                            remove_columns=dataset.column_names)

        encoded_dataset_train_inner.set_format("torch")
        encoded_dataset_test_inner.set_format("torch")

        model = AutoModelForSequenceClassification.from_pretrained("classla/bcms-bertic",
                                                                   problem_type="single_label_classification",
                                                                   num_labels=len(labels),
                                                                   id2label=id2label,
                                                                   label2id=label2id)

        # model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased",
        #                                                            problem_type="single_label_classification",
        #                                                            num_labels=len(labels),
        #                                                            id2label=id2label,
        #                                                            label2id=label2id)

        batch_size = 8
        metric_name = "f1"

        args = TrainingArguments(
            f"bert_multilabel_model",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=10,
            weight_decay=0.01,
            load_best_model_at_end=False,
            metric_for_best_model=metric_name,
            logging_steps=ceil(len(dataset_train_inner) / batch_size)
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset_train_inner,
            eval_dataset=encoded_dataset_test_inner,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics)

        trainer.train()
        metrics_train = trainer.state.log_history
        for i in range(len(metrics_train) - 1):
            if i % 2 != 0:
                f1_scores[(i - 1) // 2] += metrics_train[i]["eval_f1"]

        directory_path = 'C:/Users/Sara/PycharmProjects/bertic/bert_multilabel_model'
        folders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]

        for folder in folders:
            folder_path = os.path.join(directory_path, folder)
            shutil.rmtree(folder_path)

    f1_scores /= fold_inner_num
    best_epoch = np.argmax(f1_scores) + 1
    f1_scores_average += f1_scores

    dataset_train = Dataset.from_pandas(df_train)
    dataset_test = Dataset.from_pandas(df_test)

    encoded_dataset_train = dataset_train.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
    encoded_dataset_test = dataset_test.map(preprocess_data, batched=True, remove_columns=dataset.column_names)

    encoded_dataset_train.set_format("torch")
    encoded_dataset_test.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained("classla/bcms-bertic",
                                                               problem_type="single_label_classification",
                                                               num_labels=len(labels),
                                                               id2label=id2label,
                                                               label2id=label2id)

    # model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased",
    #                                                            problem_type="single_label_classification",
    #                                                            num_labels=len(labels),
    #                                                            id2label=id2label,
    #                                                            label2id=label2id)

    batch_size = 8
    metric_name = "f1"

    args = TrainingArguments(
        f"bert_multilabel_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=best_epoch,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model=metric_name,
        logging_steps=ceil(len(dataset_train) / batch_size),
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)

    trainer.train()
    metrics_eval = trainer.evaluate()
    f1_mean += metrics_eval["eval_f1"]
    confusion_matrices_list.append(np.array(metrics_eval["eval_confusion_matrix"]))

    directory_path = 'C:/Users/Sara/PycharmProjects/bertic/bert_multilabel_model'
    folders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]

    for folder in folders:
        folder_path = os.path.join(directory_path, folder)
        shutil.rmtree(folder_path)

f1_scores_average /= fold_num
f1_mean /= fold_num
print(f1_mean)

plt.figure()
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f1_scores_average)
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.show()

confusion_matrices_array = np.array(confusion_matrices_list)
normalized_confusion_matrices = np.zeros_like(confusion_matrices_array, dtype=float)
average_confusion_matrix = np.mean(confusion_matrices_array, axis=0)
normalized_average_confusion_matrix = average_confusion_matrix.astype(float) / average_confusion_matrix.sum(axis=1,
                                                                                                            keepdims=True)
for i in range(confusion_matrices_array.shape[0]):
    normalized_confusion_matrices[i] = confusion_matrices_array[i].astype(float) / confusion_matrices_array[i].sum(
        axis=1, keepdims=True)

confusion_matrix_variance = np.var(normalized_confusion_matrices, axis=0)

plt.figure(figsize=(8, 6))
ax = sb.heatmap(normalized_average_confusion_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                xticklabels=['Ljubavna', 'Misaono-refleksivna', 'Rodoljubiva'],
                yticklabels=['Ljubavna', 'Misaono-refleksivna', 'Rodoljubiva'])

for j in range(normalized_average_confusion_matrix.shape[0]):
    for k in range(normalized_average_confusion_matrix.shape[1]):
        if j == k:
            cell_color = ax.collections[0].get_cmap()(
                ax.collections[0].get_array().reshape(normalized_average_confusion_matrix.shape)[j, k])
            brightness = np.mean(cell_color[:3])
            text_color = 'white' if brightness < 0.6 else 'black'
            ax.text(k + 0.95, j + 0.05, f"$ \sigma $ = {np.sqrt(confusion_matrix_variance[j][k]):.2g}",
                    horizontalalignment='right', verticalalignment='top', color=text_color)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
