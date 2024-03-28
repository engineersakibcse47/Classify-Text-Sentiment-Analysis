### BERT

BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.

BERT, a popular language model, uses a special structure called the Transformer encoder. It checks each word in a sentence while considering all the other words around it. That's why it's called Bidirectional Encoder Representations from Transformers. 

### Sentiment analysis

This notebook complete code to fine-tune BERT to perform sentiment analysis to classify movie reviews as *positive* or *negative*, based on the text of the review.

Large Movie Review Dataset that contains the text of 50,000 movie reviews from the [Internet Movie Database](https://www.imdb.com/).

In this notebook:

- Load the IMDB dataset
- Load a BERT model from TensorFlow Hub
- Build model by combining BERT with a classifier
- Train model, fine-tuning BERT
- Save model and use it to classify sentences

### Setup Requirements

`conda create -p venv python ==3.10 -y`

`conda activate venv/`

`tensorflow_text-2.13.0-cp310-cp310-macosx_11_0_arm64.whl`

`pip install tensorflow-2.13.0-cp310-cp310-macosx_13_0_arm64.whl`

`pip install tensorflow_io-0.34.0-cp310-cp310-macosx_13_0_arm64.whl`

`!pip install "tf-models-official==2.13.*"`

`pip install chardet`

`conda install jupyter matplotlib`

To install tensorflow-text for mac [all relases](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases)

`jupyter notebook`

### Load the IMDB dataset
#### Train Test Validation Set

Here, I used the `text_dataset_from_directory` api to create tensor dataset `tf.data.Dataset`.

The dataset already divided into train and test, but there has no validation set. So, creating a validation set using an 80/20 split of the training data by using the `validation_split` argument.

Optimization Note:  When using the `validation_split` and `subset` arguments, make sure to either specify a random seed, or to pass `shuffle=False`, so that the validation and training splits have no overlap.

### Loading models from TensorFlow Hub

We can choose any BERT model, load from TensorFlow Hub and fine-tune. There are multiple BERT models [available](https://www.kaggle.com/models/tensorflow/bert/frameworks/tensorFlow2/variations/en-uncased-l-12-h-768-a-12/versions/3?tfhub-redirect=true).

`tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"`

`tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"`

### The preprocessing BERT model Explaination

The BERT models return 3 important keys: `pooled_output`, `sequence_output`, `encoder_outputs`:

- `pooled_output` represents each input sequence as a whole. You can think, Embedding for an entire sentence in form of numbers. The shape is `[batch_size, H]`.

- `sequence_output` represents each input token in the context. You can think of this as a contextual embedding for every token in the sentence. The shape is `[batch_size, seq_length, H]`.

- `encoder_outputs` is the encoder output of all 12 encoder layers and the last value of the list is equal to `sequence_output`.

NB: For the fine-tuning we use the `pooled_output` array.

### Define model(building functional model)

Creating a very simple fine-tuned model, with the preprocessing model, the selected BERT model, one Dense and a Dropout layer. The preprocessing model will take care of that

Functional model offers more flexibility because we don't need to attach layers in sequential order. 

`input1 = Input(shape=(X_train.shape[1],))`

`hidden1 = Dense(5, activation='sigmoid')(input1)`

`hidden2 = Dense(4, activation='sigmoid')(hidden1)`

`output = Dense(10, activation='softmax')(hidden2)`

The final step of using functional style is to initialize the entire architecture. It can be achieved using Model() function along with its parameters which defines the input and output layer.

`model_func = Model(inputs=input1, outputs=output)`

[sequential-vs-functional-model](https://becominghuman.ai/sequential-vs-functional-model-in-keras-20684f766057)


def build_classifier_model():

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

### Model training

Now, we have all the pieces to train a model, including the preprocessing module, BERT encoder, data, and classifier.

#### Loss function

Since this is a binary classification problem and the model outputs a probability (a single-unit layer), 
using `losses.BinaryCrossentropy` loss function.

`loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)`

`metrics = tf.metrics.BinaryAccuracy()`

#### Optimizer

For fine-tuning, using the same optimizer that BERT was originally trained with: the "Adaptive Moments" (Adam). This optimizer minimizes the prediction loss and does regularization by weight decay (not using moments), which is also known as [AdamW](https://arxiv.org/abs/1711.05101).

For the learning rate (`init_lr`), you will use the same schedule as BERT pre-training: linear decay of a notional initial learning rate, prefixed with a linear warm-up phase over the first 10% of training steps (`num_warmup_steps`). In line with the BERT paper, the initial learning rate is smaller for fine-tuning (best of 5e-5, 3e-5, 2e-5).

#### Plot the accuracy and loss over time

Based on the `History` object returned by `model.fit()`. You can plot the training and validation loss for comparison, as well as the training and validation accuracy:

<img width="600" alt="image" src="https://github.com/engineersakibcse47/Text-Classification--Sentiment-analysis-with-BERT/assets/108215990/6d9ab694-e768-4bf0-a69a-e9b74e9ef2df">

### Deployment with Streamlit

<img width="600" alt="image" src="https://github.com/engineersakibcse47/Text-Classification--Sentiment-analysis-with-BERT/assets/108215990/a4ecf3fc-4a3d-4fa1-bfa8-dd5c73b5c8bc">
































