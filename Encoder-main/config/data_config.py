import sys
from config import transforms_config
from config.paths_config import dataset_paths


DATASETS = {
  'deepfashion': {
    'transforms': transforms_config.EncodeTransforms,
    'train_source_root': dataset_paths['deepfashion_train'],
    'train_target_root': dataset_paths['deepfashion_train'],
    'test_source_root': dataset_paths['deepfashion_test'],
    'test_target_root': dataset_paths['deepfashion_test'],
    },
	'fashiondata': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['fashiondata_train'],
		'train_target_root': dataset_paths['fashiondata_train'],
		'test_source_root': dataset_paths['fashiondata_test'],
		'test_target_root': dataset_paths['fashiondata_test'],
	},
}
