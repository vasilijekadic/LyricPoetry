import os
import shutil
import torch
from math import ceil
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, TrainingArguments, Trainer
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sb
from matplotlib import pyplot as plt


def preprocess_data(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels

    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    mcm = multilabel_confusion_matrix(y_true, y_pred).tolist()

    metrics = {'f1': f1_macro_average,
               'confusion_matrix': mcm}

    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


df = pd.read_csv("dataset-multilabel.txt")
mlb = MultiLabelBinarizer()
mlb_result = mlb.fit_transform([str(df.loc[i, 'labels']).split(',') for i in range(len(df))])
df = pd.concat([df['text'], pd.DataFrame(mlb_result, columns=list(mlb.classes_))], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("classla/bcms-bertic")
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

labels = ['Ljubavna poezija', 'Misaono - refleksivna poezija', 'Rodoljubiva poezija']
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
encoded_dataset.set_format("torch")

fold_num = 10
fold_inner_num = 5
k_fold = IterativeStratification(n_splits=fold_num, order=1)
corpus = df["text"].copy()
y = df.loc[:, "Ljubavna poezija":"Rodoljubiva poezija"].copy()

f1_mean = 0
confusion_matrices_list = []
f1_scores_average = np.zeros(10, dtype=float)

for train, test in k_fold.split(corpus, y):
    df_train = df.loc[train].copy()
    df_test = df.loc[test].copy()
    df_train = df_train.reset_index(drop=True)

    f1_scores = np.zeros(10, dtype=float)
    k_fold_inner = IterativeStratification(n_splits=fold_inner_num, order=1)
    for train_inner, test_inner in k_fold_inner.split(df_train["text"],
                                                      df_train.loc[:, "Ljubavna poezija":"Rodoljubiva poezija"]):
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
                                                                   problem_type="multi_label_classification",
                                                                   num_labels=len(labels),
                                                                   id2label=id2label,
                                                                   label2id=label2id)

        # model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased",
        #                                                            problem_type="multi_label_classification",
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
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(labels),
                                                               id2label=id2label,
                                                               label2id=label2id)

    # model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased",
    #                                                            problem_type="multi_label_classification",
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

    ax.set_title(f'{labels[i]}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.show()
