import datasets
from datasets import Dataset
import os
from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd 
###### add_special_tokens=True
from sklearn.model_selection import train_test_split

class FormulaLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _load(self, task_name):
        df = pd.read_csv('../BERT_clean_data.csv')
        dataset=Dataset.from_pandas(df)
        print(len(dataset))
        dataset = dataset.map(partial(self.convert_to_features,tokenizer=self.tokenizer), batched=True, remove_columns='seq')
        print(dataset)
        # exit()
        train, test = dataset.train_test_split(train_size = 0.9).values()
        print(train)
        print(len(train))
        # exit()
        # train, test = dataset.train_test_split(test_size=0.2)
        # print(train)
        # print(f'Example in {split} set:')
        return train, test

    def my_load(self, task_name):
        return self._load(task_name)

    def convert_to_features(self, example_batch, tokenizer):
        # print(tokenizer)
        # tokenizer=self.tokenizer
        # print(example_batch['seq'].type)
        input_encodings = tokenizer.batch_encode_plus(example_batch['seq'], max_length=130, truncation=True, add_special_tokens=True)
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }

        return encodings


    # def my_load(self, task_name):
    #     # dataset = datasets.load_dataset('lm1b', split=split)
    #     # print(f'Example in {split} set:')
    #     # print(dataset[0])
    #     dataset = pd.read_csv('./BERT_clean_data.csv')
    #     # dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True)
    #     # print(f'Example in {split} set:')
    #     # print(dataset[0]) 
    #     train, test = train_test_split(dataset, random_state=0, train_size = 0.8,  shuffle=False)
    #     # dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True, remove_columns='text')
    #     return train, test




class DiffusionLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _load(self, task_name, split):
        dataset = datasets.load_dataset('lm1b', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True, remove_columns='text')
        return dataset

    def my_load(self, task_name, splits):
        return [self._load(task_name, name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        input_encodings = tokenizer.batch_encode_plus(example_batch['text'], max_length=128, truncation=True, add_special_tokens=True)
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }

        return encodings

class ConditionalLoader:
    def __init__(self, tokenizer, return_source_length=False):
        self.tokenizer = tokenizer
        self.return_source_length = return_source_length
        self.data_dir = './conditional_data'

    @staticmethod
    def _convert_to_features_original(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=128, truncation=True, add_special_tokens=True)
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=128, truncation=True, add_special_tokens=True)
        return {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

    def load_original(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        dataset = dataset.map(partial(self._convert_to_features_original, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def _load(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        if self.return_source_length:
            dataset = dataset.map(partial(self.add_original_src_length, tokenizer=self.tokenizer))
        dataset = dataset.map(self.add_prompt)
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def add_original_src_length(self, example, tokenizer):
        return {
            'original_src_length': len(tokenizer.encode(example['src'], max_length=128, truncation=True, add_special_tokens=True))
        }

    def my_load(self, splits):
        return [self._load(name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=128, truncation=True, add_special_tokens=True)
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=128, truncation=True, add_special_tokens=True)
        encodings = {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

        return encodings

    @staticmethod
    def collate_fn(batch_input, tokenizer):
        input_ids = pad_sequence([torch.tensor(
            [tokenizer.cls_token_id] + d['source'] + d['target'] + [tokenizer.sep_token_id]
        ) for d in batch_input], batch_first=True)

        attention_mask = torch.ones_like(input_ids)

        target_mask = torch.stack([torch.cat([
            torch.zeros(len(d['source']) + 1), torch.ones(input_ids.size(1) - len(d['source']) - 1)
        ]) for d in batch_input])

        assert input_ids.size() == attention_mask.size() == target_mask.size()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
        }

class QQPLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(QQPLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'qqp'

    @staticmethod
    def add_prompt(example):
        example['src'] = '"' + example['src'] + '" is equal to "'
        example['trg'] = example['trg']
        return example



class QTLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(QTLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'Q-T'

    @staticmethod
    def add_prompt(example):
        example['src'] = ' Answer: ' + example['src'] + ' Question: '
        example['trg'] = example['trg']
        return example


class WikiLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(WikiLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'wiki_alignment'

    @staticmethod
    def add_prompt(example):
        example['src'] = '"' + example['src'] + '" can be summarized as: '
        example['trg'] = example['trg']
        return example

class CCLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(CCLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'CC'

    @staticmethod
    def add_prompt(example):
        example['src'] = example['src'] + ' - '
        example['trg'] = example['trg']
        return example


class DiffusionLoaderWithElectra(DiffusionLoader):
    def __init__(self, model_tokenizer, electra_tokenizer, electra_model):
        super().__init__(model_tokenizer)
        self.electra_tokenizer = electra_tokenizer
        self.electra_model = electra_model

    def _load(self, task_name, split):
        dataset = datasets.load_dataset(f'./dataloaders/{task_name}.py', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        dataset = dataset.map(partial(self.new_convert_to_features, model_tokenizer=self.tokenizer, electra_tokenizer=self.electra_tokenizer, electra_model=self.electra_model), batched=True, remove_columns='text')
        return dataset

    @staticmethod
    def new_convert_to_features(example_batch, model_tokenizer, electra_tokenizer, electra_model):
        input_encodings = model_tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, add_special_tokens=True)
        electra_encodings = electra_tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, padding=True, return_tensors='pt', add_special_tokens=True)
        for k in electra_encodings.keys():
            electra_encodings[k] = electra_encodings[k].cuda()
        position = electra_encodings['attention_mask'].count_nonzero(1)
        with torch.no_grad():
            logits = electra_model(**electra_encodings)


        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'electra_logits': [logits[i][:position[i]] for i in range(position.size(0))]
        }

        return encodings



