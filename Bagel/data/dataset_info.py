# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# NOTE: The paths below are hardcoded for internal use. Replace them with your
# own data paths before running. A future release may switch to environment
# variables or a separate config file.

from .interleave_datasets import UnifiedEditIterableDataset, T2IWithReflectionThinkingIterableDataset, T2IWithReflectionThinkingV2IterableDataset, ReflectionThinkingOneConvIterableDataset, ReflectionThinkingOneConvOnlyTextIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset, SftParquetIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'vlm_sft_parquet': SftParquetIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    't2i_with_reflection_think': T2IWithReflectionThinkingIterableDataset,
    't2i_with_reflection_think_no_change': T2IWithReflectionThinkingV2IterableDataset,
    't2i_reflection_think_one_conv_balance': ReflectionThinkingOneConvIterableDataset,
    't2i_reflection_think_one_conv_balance_qwen': ReflectionThinkingOneConvIterableDataset,
    't2i_reflection_think_one_conv_balance_onlytext': ReflectionThinkingOneConvOnlyTextIterableDataset
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': './data/', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },

        't2i_bllip3o_10k': {
            'data_dir': './data/', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 10000, # number of total samples in the dataset
        },
        't2i_bllip3o_512_10k': {
            'data_dir': './data/', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 10000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': './data/',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": './data/.json', # information of the parquet files
		},

    },
    't2i_with_reflection_think': {
        "reflect_think": {
            'data_dir': './data/',
            'num_files': 2,
            'num_total_samples': 4075 * 2,
            "parquet_info_path": './data/.json', # information of the parquet files
        },
    },
    't2i_with_reflection_think_no_change': {
        "reflect_think": {
            'data_dir': './data/',
            'num_files': 2,
            'num_total_samples': 4075 * 2,
            "parquet_info_path": './data/.json', # information of the parquet files
        },
    },
    't2i_reflection_think_one_conv_balance': {
        "reflect_think": {
            'data_dir': './data/',
            'num_files': 2,
            'num_total_samples': 21905,
            "parquet_info_path": './data/.json', # information of the parquet files
        }
    },
    't2i_reflection_think_one_conv_balance_qwen': {
        "reflect_think": {
            'data_dir': './data/',
            'num_files': 5,
            'num_total_samples': 15872,
            "parquet_info_path": './data/.json', # information of the parquet files
        },
        "reflect_think_qwen_ocr": {
            'data_dir': './data/',
            'num_files': 8,
            'num_total_samples': 2197,
            "parquet_info_path": './data/.json',
        },
    },
    't2i_reflection_think_one_conv_balance_onlytext': {
        "reflect_think": {
            'data_dir': './data/',
            'num_files': 2,
            'num_total_samples': 21875,
            "parquet_info_path": './data/.json', # information of the parquet files
        }
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': './data/',
			'jsonl_path': './data/.jsonl',
			'num_total_samples': 1000
		},
    },
    'vlm_sft_parquet': {
        'llava_next': {
			'data_dir': './data/-NeXT-Data/data/',
            'num_files': 236,
            'num_total_samples': 779289,
            "parquet_info_path": './data/-NeXT-Data/parquet_info/llava-next.json', # information of the parquet files
		},
    },
}








