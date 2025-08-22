from django.views.generic import ListView
from .models import My_fnd
from django.urls import reverse_lazy
from django.views.decorators.clickjacking import xframe_options_sameorigin
from .models import My_fnd
from django.shortcuts import render
"""class Homepage(ListView):
    model = My_fnd
    template_name = "home.html"

class Output(ListView):
    model = My_fnd
    template_name = "output.html" """

from django.http import HttpResponse

@xframe_options_sameorigin
def index(request):
    # import numpy as np
    # import pandas as pd
    # import tensorflow as tf
    # import transformers
    #
    # max_length = 128  # Maximum length of input sentence to the model.
    # batch_size = 32
    # epochs = 2
    #
    # # Labels in our dataset.
    # labels = ["contradiction", "entailment", "neutral"]
    # train_df = pd.read_csv("static/SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
    # valid_df = pd.read_csv("static/SNLI_Corpus/snli_1.0_dev.csv")
    # test_df = pd.read_csv("static/SNLI_Corpus/snli_1.0_test.csv")
    # train_df.dropna(axis=0, inplace=True)
    #
    # train_df = (
    #     train_df[train_df.similarity != "-"]
    #         .sample(frac=1.0, random_state=42)
    #         .reset_index(drop=True)
    # )
    # valid_df = (
    #     valid_df[valid_df.similarity != "-"]
    #         .sample(frac=1.0, random_state=42)
    #         .reset_index(drop=True)
    # )
    #
    # train_df["label"] = train_df["similarity"].apply(
    #     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    # )
    # y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)
    #
    # valid_df["label"] = valid_df["similarity"].apply(
    #     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    # )
    # y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)
    #
    # test_df["label"] = test_df["similarity"].apply(
    #     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    # )
    # y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)
    #
    # class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    #     """Generates batches of data.
    #
    #     Args:
    #         sentence_pairs: Array of premise and hypothesis input sentences.
    #         labels: Array of labels.
    #         batch_size: Integer batch size.
    #         shuffle: boolean, whether to shuffle the data.
    #         include_targets: boolean, whether to incude the labels.
    #
    #     Returns:
    #         Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
    #         (or just `[input_ids, attention_mask, `token_type_ids]`
    #          if `include_targets=False`)
    #     """
    #
    #     def __init__(
    #             self,
    #             sentence_pairs,
    #             labels,
    #             batch_size=batch_size,
    #             shuffle=True,
    #             include_targets=True,
    #     ):
    #         self.sentence_pairs = sentence_pairs
    #         self.labels = labels
    #         self.shuffle = shuffle
    #         self.batch_size = batch_size
    #         self.include_targets = include_targets
    #         # Load our BERT Tokenizer to encode the text.
    #         # We will use base-base-uncased pretrained model.
    #         self.tokenizer = transformers.BertTokenizer.from_pretrained(
    #             "bert-base-uncased", do_lower_case=True
    #         )
    #         self.indexes = np.arange(len(self.sentence_pairs))
    #         self.on_epoch_end()
    #
    #     def __len__(self):
    #         # Denotes the number of batches per epoch.
    #         return len(self.sentence_pairs) // self.batch_size
    #
    #     def __getitem__(self, idx):
    #         # Retrieves the batch of index.
    #         indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
    #         sentence_pairs = self.sentence_pairs[indexes]
    #
    #         # With BERT tokenizer's batch_encode_plus batch of both the sentences are
    #         # encoded together and separated by [SEP] token.
    #         encoded = self.tokenizer.batch_encode_plus(
    #             sentence_pairs.tolist(),
    #             add_special_tokens=True,
    #             max_length=max_length,
    #             return_attention_mask=True,
    #             return_token_type_ids=True,
    #             pad_to_max_length=True,
    #             return_tensors="tf",
    #         )
    #
    #         # Convert batch of encoded features to numpy array.
    #         input_ids = np.array(encoded["input_ids"], dtype="int32")
    #         attention_masks = np.array(encoded["attention_mask"], dtype="int32")
    #         token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")
    #
    #         # Set to true if data generator is used for training/validation.
    #         if self.include_targets:
    #             labels = np.array(self.labels[indexes], dtype="int32")
    #             return [input_ids, attention_masks, token_type_ids], labels
    #         else:
    #             return [input_ids, attention_masks, token_type_ids]
    #
    #     def on_epoch_end(self):
    #         # Shuffle indexes after each epoch if shuffle is set to True.
    #         if self.shuffle:
    #             np.random.RandomState(42).shuffle(self.indexes)
    #
    # # Create the model under a distribution strategy scope.
    # strategy = tf.distribute.MirroredStrategy()
    #
    # with strategy.scope():
    #     # Encoded token ids from BERT tokenizer.
    #     input_ids = tf.keras.layers.Input(
    #         shape=(max_length,), dtype=tf.int32, name="input_ids"
    #     )
    #     # Attention masks indicates to the model which tokens should be attended to.
    #     attention_masks = tf.keras.layers.Input(
    #         shape=(max_length,), dtype=tf.int32, name="attention_masks"
    #     )
    #     # Token type ids are binary masks identifying different sequences in the model.
    #     token_type_ids = tf.keras.layers.Input(
    #         shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    #     )
    #     # Loading pretrained BERT model.
    #     bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    #     # Freeze the BERT model to reuse the pretrained features without modifying them.
    #     bert_model.trainable = False
    #
    #     sequence_output, pooled_output = bert_model(
    #         input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    #     )
    #     # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    #     bi_lstm = tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(64, return_sequences=True)
    #     )(sequence_output)
    #     # Applying hybrid pooling approach to bi_lstm sequence output.
    #     avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    #     max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    #     concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    #     dropout = tf.keras.layers.Dropout(0.3)(concat)
    #     output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    #     testmodel = tf.keras.models.Model(
    #         inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    #     )
    #
    #     testmodel.compile(
    #         optimizer=tf.keras.optimizers.Adam(),
    #         loss="categorical_crossentropy",
    #         metrics=["acc"],
    #     )
    #
    # print(f"Strategy: {strategy}")
    #
    # testmodel.load_weights("static/bert_weights.h5")
    #
    # def check_similarity(sentence1, sentence2):
    #     sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    #     test_data = BertSemanticDataGenerator(
    #         sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    #     )
    #
    #     proba = testmodel.predict(test_data)[0]
    #     idx = np.argmax(proba)
    #     proba = f"{proba[idx]: .2f}%"
    #     pred = labels[idx]
    #     return pred, proba
    #
    # sentence1 = "Mukesh Ambani is the richest man of world"
    # sentence2 = "Mukesh Ambani is the richest man of India"
    # res = check_similarity(sentence1, sentence2)
    # res = request.GET.get('textinp')
    # print(res)
    return render(request,"home.html")

def new_page(request):
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import transformers


    max_length = 128  # Maximum length of input sentence to the model.
    batch_size = 32
    epochs = 2

    # Labels in our dataset.
    labels = ["contradiction", "entailment", "neutral"]
    train_df = pd.read_csv("static/SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
    valid_df = pd.read_csv("static/SNLI_Corpus/snli_1.0_dev.csv")
    test_df = pd.read_csv("static/SNLI_Corpus/snli_1.0_test.csv")
    train_df.dropna(axis=0, inplace=True)

    train_df = (
        train_df[train_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
    )
    valid_df = (
        valid_df[valid_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
    )

    train_df["label"] = train_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)

    valid_df["label"] = valid_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)

    test_df["label"] = test_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)

    class BertSemanticDataGenerator(tf.keras.utils.Sequence):
        """Generates batches of data.

        Args:
            sentence_pairs: Array of premise and hypothesis input sentences.
            labels: Array of labels.
            batch_size: Integer batch size.
            shuffle: boolean, whether to shuffle the data.
            include_targets: boolean, whether to incude the labels.

        Returns:
            Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
            (or just `[input_ids, attention_mask, `token_type_ids]`
             if `include_targets=False`)
        """

        def __init__(
                self,
                sentence_pairs,
                labels,
                batch_size=batch_size,
                shuffle=True,
                include_targets=True,
        ):
            self.sentence_pairs = sentence_pairs
            self.labels = labels
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.include_targets = include_targets
            # Load our BERT Tokenizer to encode the text.
            # We will use base-base-uncased pretrained model.
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True
            )
            self.indexes = np.arange(len(self.sentence_pairs))
            self.on_epoch_end()

        def __len__(self):
            # Denotes the number of batches per epoch.
            return len(self.sentence_pairs) // self.batch_size

        def __getitem__(self, idx):
            # Retrieves the batch of index.
            indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
            sentence_pairs = self.sentence_pairs[indexes]

            # With BERT tokenizer's batch_encode_plus batch of both the sentences are
            # encoded together and separated by [SEP] token.
            encoded = self.tokenizer.batch_encode_plus(
                sentence_pairs.tolist(),
                add_special_tokens=True,
                max_length=max_length,
                return_attention_mask=True,
                return_token_type_ids=True,
                pad_to_max_length=True,
                return_tensors="tf",
            )

            # Convert batch of encoded features to numpy array.
            input_ids = np.array(encoded["input_ids"], dtype="int32")
            attention_masks = np.array(encoded["attention_mask"], dtype="int32")
            token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

            # Set to true if data generator is used for training/validation.
            if self.include_targets:
                labels = np.array(self.labels[indexes], dtype="int32")
                return [input_ids, attention_masks, token_type_ids], labels
            else:
                return [input_ids, attention_masks, token_type_ids]

        def on_epoch_end(self):
            # Shuffle indexes after each epoch if shuffle is set to True.
            if self.shuffle:
                np.random.RandomState(42).shuffle(self.indexes)

    # Create the model under a distribution strategy scope.
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Encoded token ids from BERT tokenizer.
        input_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
        # Attention masks indicates to the model which tokens should be attended to.
        attention_masks = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
        # Token type ids are binary masks identifying different sequences in the model.
        token_type_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="token_type_ids"
        )
        # Loading pretrained BERT model.
        bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        # Freeze the BERT model to reuse the pretrained features without modifying them.
        bert_model.trainable = False

        sequence_output, pooled_output = bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(sequence_output)
        # Applying hybrid pooling approach to bi_lstm sequence output.
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
        concat = tf.keras.layers.concatenate([avg_pool, max_pool])
        dropout = tf.keras.layers.Dropout(0.3)(concat)
        output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
        testmodel = tf.keras.models.Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
        )

        testmodel.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )

    print(f"Strategy: {strategy}")

    testmodel.load_weights("static/bert_weights.h5")

    def check_similarity(sentence1, sentence2):
        sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
        test_data = BertSemanticDataGenerator(
            sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
        )
        print(test_data,"fff",testmodel.predict(test_data))
        proba = testmodel.predict(test_data)[0]
        idx = np.argmax(proba)
        proba = f"{proba[idx]: .2f}%"
        pred = labels[idx]
        return pred, proba

    # sentence1 = "Mukesh Ambani is the richest man of world"
    # sentence2 = "Mukesh Ambani is the richest man of India"
    # sentence3 = request.GET.get('textinp')
    # res = check_similarity(sentence1, sentence2)
    # print(res,"lmlm")

    
        
    import requests
    sentence = request.GET.get('three')
    date_from = request.GET.get('one')
    date_to = request.GET.get('two')
    print(date_to,date_from,sentence,"here")
    if sentence=="3":
        return render(request,"output2.html")
    q = "+".join(sentence.split()) + '&'   # ye logic barabar haina ? rukh ye dekh aaya abhi output abhi kar tu joh kar rahi thi
    print(q)
    url = ('https://newsapi.org/v2/everything?'
           'q='+q+
           'from='+date_from+'&'
           'to='+date_to+'&'
           #  'sources=bbc-news,the-verge&'
           'sortBy=popularity&'
           'language=en&'
           'apiKey=ea36d96a12144e8a8d91750889efa252')
    print(url)
    response = requests.get(url)
    print(response)
    res,img = [],[]
    for i in response.json()['articles']:
        if i['source']['id'] != None:
            # print(i['source']['id'])
            print(i['title'],"||",sentence)
            # print(i['author'])
            # print(i['description'])
            # tup = check_similarity(i['title'], sentence)
            if i['urlToImage'] == None:
                i['urlToImage'] = 'https://www.pewresearch.org/wp-content/uploads/sites/8/2016/07/PJ_2016.07.07_Modern-News-Consumer_0-01.png'
            # res.append([i['title'],i['url'],i['urlToImage'],tup[0],tup[1]])
            res.append([i['title'],i['url'],i['urlToImage'],"1","2"])
            # img.append(i['urlToImage'])
            # res.append(check_similarity(i['title'], sentence))
    # print(res)
    if sentence!="3":
        return render(request,"output.html",{"res":res,"data":sentence})
    

def ifram(request):
    one=request.GET.get("one")
    two=request.GET.get("two")
    three=request.GET.get("three")
    
    return render(request,"iframeee.html",{"one":one,"two":two,"three":three})
    



