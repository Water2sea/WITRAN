# WITRAN

Our proposed method, called Water-wave Information Transmission Recurrent Acceleration Network (WITRAN), outperforms the state-of-the-art methods by 5.80% and 14.28% on long-range and ultra-long-range time series forecasting tasks respectively, as demonstrated by experiments on four benchmark datasets.

## News

Our paper, titled **WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting**, has been accepted at **NeurIPS 2023** as a **spotlight**! The final version will be released soon.

## Get Start

1. Install Python>=3.9, PyTorch 1.10.1.
2. Download data. You can obtain all the benchmark datastes from [[Autoformer](https://github.com/thuml/Autoformer)] or [[Informer](https://github.com/zhouhaoyi/Informer2020)].
3. Train the model. Please change the default dataset and parameters in `run.py` and execute it with the following command:

```bash
python run.py
```

## Citation

```bash
@inproceedings{jia2023witran,
  title={WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting},
  author={Yuxin Jia, Youfang Lin, Xinyan Hao, Yan Lin, Shengnan Guo, Huaiyu Wan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
