import argparse
import sys
from typing import Callable

import yaml

from dataset.ParallelDataset import *


def main(args):
    name_to_ds: Dict[str, Callable] = {
        'c4_gec': C4Gec,
        'fce_gec': FCE,
        'lang8': Lang8,
        'bea19': BEA19,
        'gyafc': GYAFC,
        'disco_fuse': DiscoFuse,
        'wiki_auto': WikiAuto,
        'wiki_large': WikiLarge,
        'para_bank_v2': Parabank,
        'wnc': WNC,
        'APPDIA_offensive': APPDIA,
        'Paradetox_toxic': Paradetox,
        'IteraTeR_Simplicity': IteraTeRV2_Simplicity,
        'IteraTeR_Coherence': IteraTeRV2_Coherent,
        'IteraTeR_Fluency': IteraTeRV2_Fluency,
    }
    prompts_path = args.prompts
    parallel_corpus_path = args.parallel_corpus
    prompts = yaml.load(open(prompts_path, 'r'), Loader=yaml.FullLoader)
    parallel_corpus_config = yaml.load(open(parallel_corpus_path, 'r'), Loader=yaml.FullLoader)
    read_token = parallel_corpus_config['read_token']
    write_token = parallel_corpus_config['write_token']
    datasets = parallel_corpus_config['datasets']
    for ds in datasets:
        name = ds.keys()[0]
        if ds[name]['use'] is not True:
            continue
        ds_object: ParallelDataset = name_to_ds[name](
            prompts=prompts,
            read_token=read_token,
            write_token=write_token,
            do_not_edit_prompts=prompts['do_not_edit_prompts'],
            **ds[name]
        )
        ds_object.push_to_hub()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Dataset Generator')
    parser.add_argument('--prompts', type=str, default='config/prompts.yaml', help='Prompts file path')
    parser.add_argument('--parallel_corpus', type=str, default='config/parallel_datasets.yaml',
                        help='Parallel corpus file path')
    parser.add_argument('--push_to_hub_every_corpus', type=bool, default=False, )
    args = parser.parse_args(sys.argv[1:])
    main(args)
