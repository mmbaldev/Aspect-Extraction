import pandas as pd
import xml.etree.ElementTree as et
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import datetime
from pandas.core.frame import DataFrame
import pickle
import argparse
import gzip
import json

def dataset_parser(file_path: str="../datasets/train_data.xml", type='xml') -> DataFrame:
    """
    Parse the dataset file and return a dataframe with the text and labels

    :param file_path: path to the dataset file
    :param type: type of the dataset file (xml, csv)

    :return: dataframe with the text and labels
    """
    # TODO: add support for csv
    if type == 'xml':
        xtree = et.parse(file_path)
        xroot = xtree.getroot()
        train_data = []
        for node in xroot: 
            text = node.find("text").text
            terms = []
            for aspectTerms in node.iter("aspectTerms"):
                for aspectTerm in aspectTerms:
                    terms.append(aspectTerm.attrib.get("term"))
            train_data.append({'text': text, 'labels': terms})

        return pd.DataFrame(train_data, columns = ["text", "labels"])

def json_gz_parser(file_path: str) -> DataFrame:
    """
    Parse the dataset file and return a dataframe with the text and labels

    :param file_path: path to the dataset file

    :return: dataframe with the text and labels
    """
    g = gzip.open(file_path, 'r')
    for l in g:
        yield json.loads(l)
        
def get_amazon_dataset_DF(file_path: str = '../datasets/Cell_Phones_and_Accessories.json.gz') -> DataFrame:
  i = 0
  df = {}
  for d in json_gz_parser(file_path):
    print(i, end='\r')
    df[i] = d
    i += 1
  df = pd.DataFrame.from_dict(df, orient='index', columns=['reviewText'])
  df = df.rename(columns={"reviewText": "text"})
  df['labels'] = None
  return df

def initialize_model():
    """
        Initialize the InstructABSA model and tokenizer from the huggingface model hub

        :return: tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained("kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-laptops")
    model = AutoModelForSeq2SeqLM.from_pretrained("kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-laptops")

    return tokenizer, model

def config() -> dict:
    """"
    Return the configuration for the model

    :return: BOS and EOS configuration which they are used in model input
    """

    # definition + 2 positive examples + 2 negative examples + 2 neutral examples
    BOS = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        output: features, iChat, Photobooth, garage band
        Negative example 1-
        input: Speaking of the browser, it too has problems.
        output: browser
        Negative example 2-
        input: The keyboard is too slick.
        output: keyboard
        Neutral example 1-
        input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
        output: battery
        Neutral example 2-
        input: Nightly my computer defrags itself and runs a virus scan.
        output: virus scan
        Now complete the following example-
        input: """
    
    # End of the output text
    EOS = '\noutput:'

    return {'BOS': BOS, 'EOS': EOS}

def aspects_extration(text, tokenizer=None, model=None, CONFIG=None) -> list:
    """
    Extract aspects from a given text using the InstructABSA pretrained model

    :param text: text to extract aspects from
    :param config: configuration for the model
    :param tokenizer: tokenizer for the model
    :param model: model to extract aspects from

    :return: array of aspects extracted from the text
    """
    if CONFIG is None:
        CONFIG = config()
    if tokenizer is None or model is None:
        tokenizer, model = initialize_model()
    input_tokenized = tokenizer(CONFIG['BOS'] + text + CONFIG['EOS'], return_tensors="pt").input_ids
    out_put = model.generate(input_tokenized, max_length = 128)
    return tokenizer.decode(out_put[0], skip_special_tokens=True).split(',')

def dataset_aspects_extraction(dataset_path: str,  dataset_df:DataFrame = None, df: bool=True):
    """
    Extract aspects from a given dataset using the InstructABSA pretrained model and
    clean the output by whitespace trimming and make nonaspectterms to an empty list
    and return a dataframe or a list with the text, labels and aspects extracted

    :param dataset_path: path to the dataset file
    :param df: return a dataframe or a list

    :return: dataframe or a list with the text, labels and aspects extracted
    """
    if dataset_df is None:
        dataset_df = dataset_parser(dataset_path)
    
    dataset_size = len(dataset_df)
    texts_labels_preds = []
    tokenizer, model = initialize_model()
    CONFIG = config()
    # extracting aspects from each text in the dataset
    for index, row in dataset_df.iterrows():
        print(f"Extracting Aspects For doc: {index}/{dataset_size}", end='\r')
        texts_labels_preds.append({'text': row['text'], 'labels': row['labels'], 'preds': aspects_extration(row['text'], tokenizer, model, CONFIG)})
    
    # cleaning the output
    for labels_preds in texts_labels_preds:
        if labels_preds['preds'] == ['noaspectterm']:
            labels_preds['preds'] = []
        # trimming whitespaces
        labels_preds['preds'] = [x.strip(' ') for x in labels_preds['preds']]

    if not df:
        return texts_labels_preds
    return pd.DataFrame(texts_labels_preds, columns = ["text", "labels", "preds"])

def aspects_to_csv(preds: DataFrame, csv_path: str):
    """
    Write the aspects extracted from a given dataset to a csv file

    :param preds: dataframe with the text, labels and aspects extracted
    :param csv_path: path to the csv file to write the aspects to

    :return: None
    """
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ["text", "aspects"]
        writer.writerow(headers)
        for index, row in preds.iterrows():
            writer.writerow([str(row['text'])]+row['preds'])

def list_merge_elements(list_of_lists: list):
    """
    Merge elements in a list of lists to a single list

    :param list_of_lists: list of lists

    :return: merged list
    """
    result = []
    for sublist in list_of_lists:
        result.extend(sublist)
    return result

def aggregate_aspects_S_G(preds_labels_df: DataFrame) -> (list, list):
    """
    S is the set of all aspects extracted from the text by the model and G is the set of all Golden aspects
    from the dataset itself. It is used to evaluate the model
    :param preds_labels_df: dataframe with the text, labels and aspects extracted

    :return: S and G lists
    """
    S = list_merge_elements(preds_labels_df['preds'].tolist())
    G = list_merge_elements(preds_labels_df['labels'].tolist())

    return S, G

def elements_include_in_list(element:str, list: list):
    """
    Check if an element is included in any list elements or any list elements is included an element.
    This is for the evaluation of the model, instead of checking if the element is in the list with 
    the equal operator

    :param element: apect element to check
    :param list: list of labels to check

    :return: True if the element is included in any list elements or any list elements is included an element
    """

    for x in list:
        if element in x or x in element:
            return True
        
def evaluate(preds_labels_df):
    """
    Evaluate the model using precision, recall and F1 score.
    perision = (number of common aspects between S and G) / S length
    recall = (number of common aspects between S and G) / G length
    F1 = 2*perision*recall/(perision+recall)

    :param preds_labels_df: dataframe with the text, labels and aspects extracted

    :return: precision, recall and F1 score
    """
    S, G = aggregate_aspects_S_G(preds_labels_df)
    print("S length: ", len(S))
    print("G length: ", len(G))
    common = [element for element in S if element in G]
    common_length = len(common)
    persion = common_length/len(S)
    recall = common_length/len(G)
    F1 = 2*persion*recall/(persion+recall)

    return persion, recall, F1

def dataset_statistics(dataset_df, labels=True):
    tokens_count = 0
    min_tokens = 10000000
    max_tokens = 0
    labels_count = []
    for index, row in dataset_df.iterrows():
        print(index, end='\r')
        sentence_tokens_count = len(word_tokenize(row['text']))
        tokens_count += sentence_tokens_count
        if sentence_tokens_count > max_tokens:
            max_tokens = sentence_tokens_count
        if sentence_tokens_count < min_tokens:
            min_tokens = sentence_tokens_count
        if 'labels' in row and labels:
            labels_count.append(len(row['labels']))
    print("Number of documents: ", len(dataset_df))
    print("Average number of tokens: ", tokens_count/len(dataset_df))
    print("Max number of tokens: ", max_tokens)
    print("Min number of tokens: ", min_tokens)
    if labels:
        print("Average number of labels: ", sum(labels_count)/len(labels_count))
        print("Max number of labels: ", max(labels_count))
        print("Min number of labels: ", min(labels_count))
        print("Number of documents with no labels: ", labels_count.count(0))

    # Plot top 15 aspects
    if 'labels' in dataset_df:
        all_labels = [label for sublist in dataset_df['labels'] for label in sublist]
        label_counts = pd.Series(all_labels).value_counts()
        top_15_labels = label_counts.head(15)
        plt.figure(figsize=(10, 6))
        top_15_labels.plot(kind='bar', color='skyblue')
        plt.title('Top 15 Aspects Distributions')
        plt.xlabel('Aspects')
        plt.ylabel('Frequency')
        plt.show()

def main():

    # paring the command line arguments
    parser = argparse.ArgumentParser(description='A Python program to extract aspects from a given text')
    parser.add_argument('-text', required=False, help='Enter the text:')
    parser.add_argument('-dataset', required=False, help='Enter the path of xml dataset:')
    parser.add_argument('-csv', required=False, help='Enter the path of aspects output csv:')
    parser.add_argument('--evaluate', action='store_true', help='Return the evaluation of the model on the dataset')


    args = parser.parse_args()

    # inference the model for a given text
    if args.text:
        print(aspects_extration(args.text))

    # run the model on a dataset and save the output to a csv file
    elif args.dataset:
        print("start time: ", datetime.datetime.now())

        # calculating the aspects for the dataset
        test_data_labeld_preds = dataset_aspects_extraction(args.dataset)

        # save the output pickle for debugging purposes        
        with open(f'{args.dataset}.pickle', 'wb') as f:
            pickle.dump(test_data_labeld_preds, f)
        
        # save the csv aspects predictions
        if args.csv:
            aspects_to_csv(test_data_labeld_preds, args.csv)
        
        # evaluate the model
        if args.evaluate:
            persion, recall, F1 = evaluate(test_data_labeld_preds)
            print("persion: ", persion, " recall: ", recall, " F1: ", F1)

        print("end time: ", datetime.datetime.now())

    else:
        amazon_dataset_df = get_amazon_dataset_DF('./datasets/Cell_Phones_and_Accessories_5.json.gz')
        preds_df = dataset_aspects_extraction(None, amazon_dataset_df[:200])
        print(preds_df)
        aspects_to_csv(preds_df, "amazon_predictions_first_200.csv")

if __name__ == "__main__":
    main()