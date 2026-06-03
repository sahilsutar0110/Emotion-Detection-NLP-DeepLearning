import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Load Dataset
# =========================

train_df = pd.read_csv(
    "dataset/train.txt",
    sep=";",
    names=["text", "emotion"]
)

test_df = pd.read_csv(
    "dataset/test.txt",
    sep=";",
    names=["text", "emotion"]
)

val_df = pd.read_csv(
    "dataset/val.txt",
    sep=";",
    names=["text", "emotion"]
)

# =========================
# Text Cleaning
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)
val_df["text"] = val_df["text"].apply(clean_text)

# =========================
# Label Encoding
# =========================

encoder = LabelEncoder()

train_df["emotion"] = encoder.fit_transform(train_df["emotion"])
test_df["emotion"] = encoder.transform(test_df["emotion"])
val_df["emotion"] = encoder.transform(val_df["emotion"])

# =========================
# Tokenization
# =========================

tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_df["text"])

X_train = tokenizer.texts_to_sequences(train_df["text"])
X_test = tokenizer.texts_to_sequences(test_df["text"])
X_val = tokenizer.texts_to_sequences(val_df["text"])

# =========================
# Padding
# =========================

MAX_LENGTH = 100

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post"
)

X_val = pad_sequences(
    X_val,
    maxlen=MAX_LENGTH,
    padding="post"
)

# =========================
# Labels
# =========================

y_train = train_df["emotion"].values
y_test = test_df["emotion"].values
y_val = val_df["emotion"].values

# =========================
# Function for train.py
# =========================

def get_data():
    return (
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        tokenizer,
        encoder
    )

# =========================
# Debug Section
# =========================

if __name__ == "__main__":

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("X_val  :", X_val.shape)

    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)
    print("y_val  :", y_val.shape)

    print("\nVocabulary Size:", len(tokenizer.word_index))

    print("Max token value:", X_train.max())
    print("Min token value:", X_train.min())

    print("\nFirst 10 Labels:")
    print(y_train[:10])

    print("\nClass Distribution:")
    import numpy as np
    print(np.bincount(y_train))

    print("\nFirst 5 texts:")
    print(train_df["text"].head())

    print("\nFirst 5 labels:")
    print(train_df["emotion"].head())


