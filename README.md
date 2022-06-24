# Kaggle数字识别竞赛

[Kaggle竞赛](https://www.kaggle.com/competitions/digit-recognizer/overview)，简单mnist数字识别。

日期: 2022.6.24

作者: zhhike

github主页: [here](https://github.com/zh-hike)

## 环境
```
conda create -n zh python=3.9
conda activate zh
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

## 运行
* 数据准备

去kaggle下载数据，记录数据存放位置，例如 `/data`

* 训练(linux环境)
```commandline
bash run.sh
```

## 结果
* 排名: 854/1443
* score: 0.9798
