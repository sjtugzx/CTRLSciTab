from argparse import ArgumentParser
import json
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
# import keras.callbacks as callbacks

def tsdae_train(train_sentences):
    model_name = 'bert-base-uncased'
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name,
                                                 tie_encoder_decoder=True)

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True,

        # callback=callbacks.EarlyStopping(monitor='loss',min_delta=0.002,patience=0,mode='auto',restore_best_weights=False)
    )

    model.save('output/tsdae_new-model')

def generate_training_data(file_list):
    candidate_sentences = []
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("length of data: ", len(data))
            for d in range(len(data)):
                text = cell_separator.join(data[d]['table_highlight_cell']) + " " + highlight_separator + " "
                text = row_seperator + ' ' + cell_separator
                row_len = len(data[d]['table_column_names'])
                for i, c in enumerate(data[d]['table_column_names']):
                    text += ' ' + c
                    if i < row_len - 1:
                        text += ' ' + cell_separator

                # for row in data[d]['table_content_values']:
                #     text += ' ' + row_seperator + ' ' + cell_separator
                #     for i, c in enumerate(row):
                #         text += ' ' + c
                #         if i < row_len - 1:
                #             text += ' ' + cell_separator

                # text += ' ' + highlight_separator + data[d]['highlight_cells']

                initial_text = text
                background_sentences = data[d]['background_information'].split('.')


                for sentence in background_sentences:
                    candidate_text = initial_text+ ' ' + caption_separator + ' ' + data[d][
                        'table_caption'] + ' ' + background_separator + ' ' + sentence + '\n'

                    candidate_sentences.append(candidate_text)

                descp = data[d]['text'].replace('[CONTINUE]', '') + '\n'

                reference_text = initial_text + ' ' + caption_separator + ' ' + data[d]['table_caption'] + \
                                         ' ' + background_separator + ' ' + descp + '\n'
                candidate_sentences.append(reference_text)

    return candidate_sentences


if __name__ == "__main__":

    row_seperator = '<R>'
    cell_separator = '<C>'
    caption_separator = '<CAP>'
    background_separator = '<BKG>'
    highlight_separator = '<H>'

    file_lsit = [
        '../../MyDataset/dev.json',
        '../../MyDataset/test.json',
        '../../MyDataset/train.json',
    ]
    train_sentences = generate_training_data(file_lsit)
    tsdae_train(train_sentences)