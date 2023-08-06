from typing import Iterator, Tuple, Any

import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import torch
from PIL import Image


class UCSDPickAndPlace(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for UCSD Pick and Place dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x gripper position,'
                                '3x gripper orientation, 1x finger distance].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(4,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x gripper velocities,'
                            '1x gripper open/close torque].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'n_transitions': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Number of transitions in the episode.'
                    ),
                    'success': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if the last state of an episode is '
                            'a success state, False otherwise.'
                    ),
                    'success_labeled_by': tfds.features.Text(
                        doc='Who labeled success (and thereby reward) of the '
                            'episode. Can be one of: [human, classifier].'
                    ),
                    'disclaimer': tfds.features.Text(
                        doc='Disclaimer about the particular episode.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data
            data = np.load(episode_path, allow_pickle=True)
            # note: there is 1 more observation than action/reward
            ep_len = len(data['action'])

            # special instruction + reward handling for the green object
            green_object_idxs = [549, 550, 558, 559, 560, 561, 562]
            idx = int(os.path.splitext(os.path.basename(episode_path))[0])

            # compute Kona language embedding
            if 'sink' in episode_path:
                language_instruction = 'place the pot in the sink'
            elif 'tabletop_base' in episode_path:
                language_instruction = 'pick up the red object from the table'
            elif 'tabletop_uncurated' in episode_path:
                if idx in green_object_idxs:
                    language_instruction = 'pick up the green object from the table'
                else:
                    language_instruction = 'pick up the red object from the table'
            else:
                raise ValueError('Unknown episode path: {}'.format(episode_path))
            language_embedding = self._embed([language_instruction])[0].numpy()
            
            # convert to float32
            data['state'] = data['state'].astype(np.float32)
            data['action'] = data['action'].astype(np.float32)
            data['reward'] = data['reward'].astype(np.float32)

            # make label description human-readable
            if 'labeled' in data and isinstance(data['labeled'], int):
                data['labeled'] = 'human'
            else:
                data['labeled'] = 'classifier'

            # reward is unreliable for the green object
            if 'tabletop_uncurated' in episode_path and idx in green_object_idxs:
                data['reward'] *= 0
                data['disclaimer'] = 'no successful episodes for this object'
            elif 'tabletop_uncurated' in episode_path:
                data['disclaimer'] = 'reward is noisy for this dataset partition'
            else:
                data['disclaimer'] = 'none'

            def get_image(episode_path, idx):
                """Images are stored as files, so we load them from disk."""
                fp = os.path.dirname(episode_path) + data['observation'][i]
                with Image.open(fp) as im:
                    obs = np.asanyarray(im.convert("RGB"))
                return obs

            episode = []
            for i in range(ep_len):
                episode.append({
                    'observation': {
                        'image': get_image(episode_path, i),
                        'state': data['state'][i,:7]/100.,
                    },
                    'action': data['action'][i],
                    'discount': 1.0,
                    'reward': data['reward'][i],
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'n_transitions': ep_len,
                    'success': bool(data['success']),
                    'success_labeled_by': data['labeled'],
                    'disclaimer': data['disclaimer'],
                }
            }

            return episode_path, sample

        # path to data
        path = 'data/train/*/*.p'

        # create list of all examples
        episode_paths = glob.glob(path)
        print('Found {} episodes.'.format(len(episode_paths)))

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)
