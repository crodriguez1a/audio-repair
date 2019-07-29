# audio-repair
Nearest Neighbor Imputation for Damaged Audio

## Installing Dependencies

```
pipenv sync
```

## Data

Wav data can be found in [S3](https://capstone4audacity.s3.amazonaws.com/audio_sources.zip).

Saved and pre-processed Numpy array checkpoints can be found in the [checkpoints directory](https://github.com/crodriguez1a/audio-repair/tree/capstone/notebooks/submission/checkpoints)

## Classes and Utils

- [`AudioSegment`](https://github.com/crodriguez1a/audio-repair/blob/capstone/notebooks/submission/audio_segment.py)

- [`Clusterer`](https://github.com/crodriguez1a/audio-repair/blob/capstone/notebooks/submission/clusterer.py)

- [visualizers](https://github.com/crodriguez1a/audio-repair/blob/capstone/notebooks/submission/visualizers.py)

## Third-party Libraries

- [Librosa](https://librosa.github.io/librosa/)
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)
- [Impyute](https://github.com/eltonlaw/impyute/)
