import difflib
import random
import re
from random import sample
from typing import List, Dict

import datasets
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset

split_seed = 443


class ParallelDataset:
    generated_ds: DatasetDict

    def __init__(self, **kwargs):
        self.ds_name = kwargs.get('name')
        self.main_dataset = load_dataset(self.ds_name, token=kwargs.get('read_token'))
        self.repeat_with_different_prompts = kwargs.get('repeat_with_different_prompts', 1)
        self.mask_sample_size = kwargs.get('mask_sample_size', 4)
        self.no_edit_prompts = kwargs.get('do_not_edit_prompts')
        self.write_token = kwargs.get('write_token')
        self.max_no_edit_samples = kwargs.get('max_no_edit_samples', 5000)
        self.diff_mask = kwargs.get('diff_mask', False)
        self.preprocess_dataset()

    def generate_dataset(self):
        raise NotImplementedError('generate_dataset() is not implemented')

    def preprocess_dataset(self):
        pass

    def push_to_hub(self):
        self.generated_ds.push_to_hub(
            f'{self.ds_name}_aug',
            private=True,
            token=self.write_token
        )

    def get_dataset(self, max_train_size=None):
        if max_train_size is not None and 'train' in self.generated_ds.keys():
            self.generated_ds['train'] = self.generated_ds['train'].select(range(max_train_size))
        return self.generated_ds

    def generate_no_edit_dataset(self, input_column, ds, ds_name):
        simple_ds = ds.map(
            lambda x: self.create_simple_pair(self.no_edit_prompts, x[input_column], x[input_column], times=1),
            batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, ds_name, 'no edit')
        last_mask_sample_size = self.mask_sample_size
        self.mask_sample_size = 2
        mask_ds = ds.map(
            lambda x: self.create_masked_pair(self.no_edit_prompts, x[input_column], x[input_column], diff_mask=False),
            batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, ds_name, 'no edit')
        self.mask_sample_size = last_mask_sample_size
        ds = concatenate_datasets([simple_ds, mask_ds]).shuffle()
        return ds.select(range(min(len(ds), self.max_no_edit_samples)))

    @staticmethod
    def add_type_to_dataset(dataset, dataset_name, type_name):
        return dataset.add_column('type', [type_name] * len(dataset)) \
            .add_column('from', [dataset_name] * len(dataset))

    def create_simple_pair(
            self,
            prompts: List[str],
            inputs: List[str],
            outputs: List[str],
            times=None
    ) -> Dict[str, List[str]]:
        times = self.repeat_with_different_prompts if times is None else times
        if times > len(prompts):
            times = len(prompts)
        sampled_prompts = []
        for i in range(len(inputs)):
            sampled_prompts.append(sample(prompts, times))
        new_inputs = [''] * len(inputs) * times
        for i, text in enumerate(inputs):
            for j in range(times):
                new_inputs[i + len(inputs) * j] = f'{sampled_prompts[i][j]}: {text} ->'
        return {'input': new_inputs, 'output': outputs * times}

    def create_masked_pair(
            self,
            prompts: List[str],
            inputs: List[str],
            outputs: List[str],
            random_mask=True,
            diff_mask=None,
            dont_mask_words=None
    ) -> Dict[str, List[str]]:
        if diff_mask is None:
            diff_mask = self.diff_mask
        generated_inputs = []
        generated_outputs = []
        for i in range(len(inputs)):
            input_text = inputs[i]
            output_text = outputs[i]
            masked_texts, masked_texts_outputs = self.get_mask_output(
                input_text,
                output_text,
                diff_mask=diff_mask,
                random_mask=random_mask,
                dont_mask_words=dont_mask_words[i] if dont_mask_words is not None else None
            )
            for masked_text, masked_text_output in zip(masked_texts, masked_texts_outputs):
                prompt = sample(prompts, 1)[0]
                generated_inputs.append(f'{prompt}: {input_text} -> {masked_text}')
                generated_outputs.append(masked_text_output)

        return {'input': generated_inputs, 'output': generated_outputs}

    def get_mask_output(
            self,
            input_text,
            output_text,
            diff_mask=True,
            random_mask=True,
            dont_mask_words=None
    ):
        input_list = input_text.split()
        output_list = output_text.split()
        masked_texts_list = []

        if diff_mask:
            matcher = difflib.SequenceMatcher(None, output_list, input_list)
            differences = list(matcher.get_opcodes())
            masked_output = output_list.copy()
            for tag, i1, i2, j1, j2 in differences:
                if tag != 'equal':
                    for i in range(i1, i2):
                        masked_output[i] = '[MASK]'
            masked_texts_list.append(masked_output)

            mask_indices = [i for i, word in enumerate(masked_output) if word == '[MASK]']
            for j in range(2):
                if len(mask_indices) == j:
                    break
                chosen_mask_index = random.choice(mask_indices)
                neighbors = [chosen_mask_index]
                for i in range(chosen_mask_index - 1, -1, -1):
                    if masked_output[i] == '[MASK]':
                        neighbors.append(i)
                    else:
                        break
                for i in range(chosen_mask_index + 1, len(masked_output)):
                    if masked_output[i] == '[MASK]':
                        neighbors.append(i)
                    else:
                        break

                new_masked_output = output_list.copy()
                for i in neighbors:
                    new_masked_output[i] = '[MASK]'
                masked_texts_list.append(new_masked_output)
        if random_mask:
            for j in range(self.mask_sample_size):
                if len(output_list) == 0:
                    break
                masked_output = output_list.copy()
                span_nums = random.randint(1, max(min(len(output_list) // 10, 10), 2))
                for i in range(span_nums):
                    span_length = random.randint(1, min(10, len(output_list)))
                    span_start = random.randint(0, len(output_list) - span_length)
                    for j in range(span_start, span_start + span_length):
                        masked_output[j] = '[MASK]'
                masked_texts_list.append(masked_output)

        if dont_mask_words is not None:
            for i in range(len(masked_texts_list)):
                for j in dont_mask_words:
                    masked_texts_list[i][j] = output_list[j]
        masked_text = []
        masked_text_output = []
        for masked_text_list in masked_texts_list:
            extra_id_iterator = 0
            output = ''
            last_word_was_mask = False
            first_extra_id_added = False
            for i in range(len(masked_text_list)):
                word = masked_text_list[i]
                if word == '[MASK]':
                    if last_word_was_mask:
                        masked_text_list[i] = ''
                    else:
                        last_word_was_mask = True
                        masked_text_list[i] = f'<extra_id_{extra_id_iterator}>'
                        extra_id_iterator += 1
                        if not first_extra_id_added:
                            output = '<extra_id_0>'
                            first_extra_id_added = True
                    output += f' {output_list[i]}'
                else:
                    if last_word_was_mask:
                        output += f' <extra_id_{extra_id_iterator}>'
                        last_word_was_mask = False
            if last_word_was_mask:
                output += f' <extra_id_{extra_id_iterator}>'
            masked_text.append(' '.join(masked_text_list))
            masked_text_output.append(output)
        return masked_text, masked_text_output


class C4Gec(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['gec_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train']
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'c4_gec')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']), batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'c4_gec', 'simple')
        mask_ds = train_ds.map(lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']), batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'c4_gec', 'mask')
        return DatasetDict({'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()})

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(
            lambda x: len(x['input'].split()) > 10 and len(x['input'].split()) < 100 and not (
                    len(x['input']) > 1.3 * len(x['output']) or len(x['input']) < 0.7 * len(x['output']))
        )


class FCE(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['gec_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        ds = self.main_dataset
        train_ds = datasets.concatenate_datasets([ds['train'], ds['validation'], ds['test']])
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'fce_gec')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'fce_gec', 'simple')
        mask_ds = train_ds.map(lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
                               batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'fce_gec', 'mask')
        return DatasetDict({'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()})

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(
            lambda x: x['still_need_edit'] == [] and len(x['text'].split()) < 300)
        self.main_dataset = self.main_dataset.remove_columns(['still_need_edit'])
        self.main_dataset = self.main_dataset.rename_columns({'text': 'input', 'edited_text': 'output'})


class Lang8(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['gec_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train']
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'lang8_gec')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'lang8_gec', 'simple')
        mask_ds = train_ds.map(lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
                               batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'lang8_gec', 'mask')
        return DatasetDict({'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()})

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(lambda x: len(x['text'].split()) < 300)
        self.main_dataset = self.main_dataset.rename_columns({'text': 'input', 'edited_text': 'output'})


class BEA19(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['gec_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        ds = self.main_dataset
        train_ds = datasets.concatenate_datasets([ds['train'], ds['validation']])
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'bea19_gec')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'bea19_gec', 'simple')
        mask_ds = train_ds.map(lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
                               batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'bea19_gec', 'mask')
        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(
            lambda x: len(x['text'].split()) < 350 and x['still_need_edit'] == [])
        self.main_dataset = self.main_dataset.remove_columns(['still_need_edit'])
        self.main_dataset = self.main_dataset.rename_columns({'text': 'input', 'edited_text': 'output'})


class GYAFC(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['formality_prompts']
        self.reverse_prompts = prompts['formality_rev_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        ds = self.main_dataset
        train_ds = datasets.concatenate_datasets([ds['train'], ds['validation'], ds['test']])
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'gyafc_formal')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'gyafc_formal', 'simple')
        mask_ds = train_ds.map(
            lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
            batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'gyafc_formal', 'mask')

        simple_ds_rev = train_ds.map(
            lambda x: self.create_simple_pair(self.reverse_prompts, x['output'], x['input']),
            batched=True)
        simple_ds_rev = self.add_type_to_dataset(simple_ds_rev, 'gyafc_informal', 'simple')
        mask_ds_rev = train_ds.map(
            lambda x: self.create_masked_pair(self.reverse_prompts, x['output'], x['input']),
            batched=True
        )
        mask_ds_rev = self.add_type_to_dataset(mask_ds_rev, 'gyafc_informal', 'mask')

        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, simple_ds_rev, mask_ds_rev, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(lambda x: len(x['input_text'].split()) > 8)
        self.main_dataset = self.main_dataset.rename_columns({'input_text': 'input', 'output_text': 'output'})


class DiscoFuse(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['coherence_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train']
        ds_split = train_ds.train_test_split(train_size=0.95, seed=split_seed)
        train_ds = ds_split['train']
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'disco_fuse_coh')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'disco_fuse_coh', 'simple')
        mask_ds = train_ds.map(
            lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
            batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'disco_fuse_coh', 'mask')
        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(lambda x: len(x['incoherent_first_sentence'].split()) > 8 and len(
            x['incoherent_second_sentence'].split()) > 8)
        self.main_dataset = self.main_dataset.map(lambda example: {
            'input': example['incoherent_first_sentence'] + ' ' + example['incoherent_second_sentence']
        })
        self.main_dataset = self.main_dataset.rename_column('coherent_first_sentence', 'output')
        self.main_dataset = self.main_dataset.remove_columns(
            ['connective_string', 'discourse_type', 'coherent_second_sentence', 'has_coref_type_pronoun',
             'incoherent_first_sentence', 'incoherent_second_sentence', 'has_coref_type_nominal'])


class WikiAuto(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['simplification_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train']
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'wiki_auto_simplicity')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'wiki_auto_simplicity', 'simple')
        mask_ds = train_ds.map(
            lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
            batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'wiki_auto_simplicity', 'mask')
        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(
            lambda x: 10 < len(x['normal'].split()) < 500 and 30 < len(x['simple'].split()) < 500)
        self.main_dataset = self.main_dataset.rename_columns({'normal': 'input', 'simple': 'output'})


class WikiLarge(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['simplification_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train']
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'wiki_large_simplicity')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'wiki_large_simplicity', 'simple')
        mask_ds = train_ds.map(
            lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
            batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'wiki_large_simplicity', 'mask')
        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(
            lambda x: 20 < len(x['input'].split()) < 100 and 15 < len(x['simple'].split()) < 50)
        self.main_dataset = self.main_dataset.rename_columns({'simple': 'output'})


class Parabank(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['paraphrasing_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train']
        train_ds = train_ds.train_test_split(train_size=0.95, seed=split_seed)['train']
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'parabank_paraphrase')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'parabank_paraphrase', 'simple')
        mask_ds = train_ds.map(
            lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
            batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'parabank_paraphrase', 'mask')
        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.remove_columns(['similarity'])
        self.main_dataset = self.main_dataset.rename_columns({'input_text': 'input', 'paraphrased_text': 'output'})


class WNC(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['neutralization_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train']
        train_ds = train_ds.train_test_split(train_size=0.95, seed=split_seed)['train']
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'wnc_neut')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'wnc_neut', 'simple')
        mask_ds = train_ds.map(lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
                               batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'wnc_neut', 'mask')
        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.rename_columns({'text': 'input', 'edited_text': 'output'})


class APPDIA(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['offensive_prompts']
        self.reverse_prompts = prompts['offensive_rev_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = concatenate_datasets([self.main_dataset['train'], self.main_dataset['validation']])
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'appdia_offensive')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'appdia_offensive', 'simple')
        mask_ds = train_ds.map(lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
                               batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'appdia_offensive', 'mask')
        simple_ds_rev = train_ds.map(lambda x: self.create_simple_pair(self.reverse_prompts, x['output'], x['input']),
                                     batched=True)
        simple_ds_rev = self.add_type_to_dataset(simple_ds_rev, 'appdia_offensive_rev', 'simple')
        mask_ds_rev = train_ds.map(lambda x: self.create_masked_pair(self.reverse_prompts, x['output'], x['input']),
                                   batched=True)
        mask_ds_rev = self.add_type_to_dataset(mask_ds_rev, 'appdia_offensive_rev', 'mask')

        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, mask_ds_rev, simple_ds_rev, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.rename_columns(
            {'offensive-text': 'input', 'style-transferred-text': 'output'})


class Paradetox(ParallelDataset):
    def __init__(self, prompts, **kwargs):
        super().__init__(**kwargs)
        self.prompts = prompts['toxic_prompts']
        self.reverse_prompts = prompts['toxic_rev_prompts']
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train'].train_test_split(train_size=0.95)['train']

        ds1 = train_ds.remove_columns(['neutral2', 'neutral3']).rename_columns({'neutral1': 'output'}).filter(
            lambda x: x['output'] is not None)
        ds2 = train_ds.remove_columns(['neutral1', 'neutral3']).rename_columns({'neutral2': 'output'}).filter(
            lambda x: x['output'] is not None)
        ds3 = train_ds.remove_columns(['neutral1', 'neutral2']).rename_columns({'neutral3': 'output'}).filter(
            lambda x: x['output'] is not None)
        train_ds = concatenate_datasets([ds1, ds2, ds3])
        no_edit_ds = self.generate_no_edit_dataset('input', train_ds, 'paradetox_toxic')
        simple_ds = train_ds.map(lambda x: self.create_simple_pair(self.prompts, x['input'], x['output']),
                                 batched=True)
        simple_ds = self.add_type_to_dataset(simple_ds, 'paradetox_toxic', 'simple')
        mask_ds = train_ds.map(lambda x: self.create_masked_pair(self.prompts, x['input'], x['output']),
                               batched=True)
        mask_ds = self.add_type_to_dataset(mask_ds, 'paradetox_toxic', 'mask')
        simple_ds_rev = train_ds.map(lambda x: self.create_simple_pair(self.reverse_prompts, x['output'], x['input']),
                                     batched=True)
        simple_ds_rev = self.add_type_to_dataset(simple_ds_rev, 'paradetox_toxic_rev', 'simple')
        mask_ds_rev = train_ds.map(lambda x: self.create_masked_pair(self.reverse_prompts, x['output'], x['input']),
                                   batched=True)
        mask_ds_rev = self.add_type_to_dataset(mask_ds_rev, 'paradetox_toxic_rev', 'mask')

        return DatasetDict({
            'train': concatenate_datasets([simple_ds, mask_ds, mask_ds_rev, simple_ds_rev, no_edit_ds]).shuffle()
        })

    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(
            lambda x: len(x['toxic'].split()) > 6 and len(x['neutral1'].split()) > 6)
        self.main_dataset = self.main_dataset.rename_column('toxic', 'input')


class IteraTeRV2(ParallelDataset):
    def __init__(self, task_type, prompts, **kwargs):
        self.type = task_type
        self.pattern = r'<S>(.*?)</S>'
        self.prompts = prompts
        super().__init__(**kwargs)
        self.generated_ds = self.generate_dataset()

    def generate_dataset(self) -> DatasetDict:
        train_ds = self.main_dataset['train']
        train_ds = train_ds.map(lambda x: self.generate_pairs(x['input'], x['output']), batched=True)
        self.add_type_to_dataset(train_ds, 'IteraTeRV2', self.type)
        return DatasetDict({'train': train_ds})

    def generate_pairs(self, before_sentences: List[str], after_sentences: List[str]):
        intent = f'<{self.type}>'

        inputs: List[str] = []
        outputs: List[str] = []
        for i in range(len(before_sentences)):
            before_sentence = before_sentences[i]
            output = after_sentences[i]
            if not before_sentence.startswith(intent):
                return {}
            before_sentence = before_sentence[len(intent):]
            after_sentence = re.sub(self.pattern, '<extra_id_0>', before_sentence)
            output = f'<extra_id_0> {output} <extra_id_1>'
            before_sentence = before_sentence.replace('<S>', '').replace('</S>', '')
            for _ in range(self.repeat_with_different_prompts):
                inputs.append(f'{random.sample(self.prompts, 1)[0]}: {before_sentence} -> {after_sentence}')
                outputs.append(output)
        return {'input': inputs, 'output': outputs}


    def preprocess_dataset(self):
        self.main_dataset = self.main_dataset.filter(lambda x: x['labels'] == self.type)
        self.main_dataset = self.main_dataset.filter(
            lambda x: len(x['before_sent_with_intent'].split()) > 10 and len(x['after_sent'].split()) > 10 and len(
                re.findall(self.pattern, x['before_sent_with_intent'])) == 1)
        self.main_dataset = self.main_dataset.remove_columns(
            ['before_sent', 'labels', 'confidence', 'doc_id', 'revision_depth'])
        self.main_dataset = self.main_dataset.rename_columns({'before_sent_with_intent': 'input', 'after_sent': 'output'})

    def push_to_hub(self):
        self.generated_ds.push_to_hub(
            f'{self.ds_name}_{self.type}_aug',
            private=True,
            token=self.write_token
        )


class IteraTeRV2_Simplicity(IteraTeRV2):
    def __init__(self, prompts, **kwargs):
        super().__init__(task_type='clarity', prompts=prompts['simplification_prompts'], **kwargs)


class IteraTeRV2_Coherent(IteraTeRV2):
    def __init__(self, prompts, **kwargs):
        super().__init__(task_type='coherence', prompts=prompts['coherence_prompts'], **kwargs)


class IteraTeRV2_Fluency(IteraTeRV2):
    def __init__(self, prompts, **kwargs):
        super().__init__(task_type='fluency', prompts=prompts['gec_prompts'], **kwargs)
