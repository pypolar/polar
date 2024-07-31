# PointNet AutoEncoder training on ModelNet40


Training script can be found in `polar.train.train.py` with the function `main`. It can be imported from `polar` as `train_ae`.

## Minimal example
Typically, a training would be done as follows:

1. Create a file `train_ae.py` with the following:
```python
from polar import train_ae

if __name__ == '__main__':
    train_ae()
```
2. Then, run

```
python train_ae.py --name demo --shuffle --sigma 0.05
```

## Parameters
It accepts the following parameters:

- **Base**
    - `name` (*Required*)
    - `log_dir` (`str`, default=`'logs/ae'`)
    - `batch_size` (`int`, default=`64`)
    - `num_workers` (`int`, default=`4`)

- **Dataset** 
    - `rootdir` (`str`, default=`'modelnet'`)
    - `classes` (`str`, default=`None`)
    - `exclude_classes` (`str`, default=`None`)
    - `samples_per_class` (`int`, default=`None`)

- **Preprocessing**
    - `shuffle` (`bool`, default=`False`)
    - `num_points` (`int`, default=`1024`)
    - `max_angle` (`int`, default=`180`)
    - `max_trans` (`float`, default=`0.0`)

- **Augmentations**
    - `sigma` (`float`, default=`0.0`)
    - `min_scale` (`float`, default=`1.0`)
    - `keep_ratio` (`float`, default=`1.0`)
    - `p` (`float`, default=`0.5`)

- **Autoencoder**
    - `first_stage_widths` (`int`, default=`(64, 64)`)
    - `second_stage_widths` (`int`, default=`(64, 128, 1024)`)
    - `decoder_widths` (`int`, default=`(1024, 1024)`)
    - `dropout` (`float`, default=`0.1`)

- **Training**
    - `lr` (`float`, default=`0.001`)
    - `resume_optimizer` (`bool`, default=`False`)
    - `checkpoint` (`str`, default=`None`)
    - `freeze_decoder` (`bool`, default=`False`)
    - `epochs` (`int`,  default=`150`)

- **Loss**
    - `norm` (`int`, default=`2`)
    - `density_weight` (`float`, default=`0.0`)
    - `density_radius` (`float`, default=`0.1`)
