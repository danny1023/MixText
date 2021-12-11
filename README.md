# MixText4CN

以下论文是基于英文语料进行训练测试的，本项目主要基于该论文代码，将语料修改为中文。

*Jiaao Chen, Zichao Yang, Diyi Yang*: MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification. In Proceedings of the 58th Annual Meeting of the Association of Computational Linguistics (ACL'2020)


## 开始

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* Pytorch_transformers (also known as transformers)
* Pandas, Numpy, Pickle
* Fairseq


### 代码结构
```
|__ data/
        |__ yahoo_answers_csv/ --> 雅虎问答数据集
            |__ back_translate.ipynb --> 反向翻译的数据集
            |__ classes.txt --> 雅虎问答数据的类别
            |__ train.csv --> 原始训练集
            |__ test.csv --> 原始测试集
            |__ de_1.pkl --> 使用德语作为中间语言的反向翻译
            |__ ru_1.pkl --> 使用俄语作为中间语言的反向翻译

|__code/
        |__ transformers/ --> huggingface/transformers 的代码
        |__ read_data.py --> 读取数据集的代码， 组成有标签数据集，无标签数据集，forming labeled training set, unlabeled training set, development set and testing set; building dataloaders
        |__ normal_bert.py --> BERT baseline模型
        |__ normal_train.py --> 训练BERT baseline模型
        |__ mixtext.py --> TMix/MixText 模型代码
        |__ train.py --> 训练和测试 TMix/MixText 模型

|__model/
```

### 下载数据
下载数据放到data目录下. You can find Yahoo Answers, AG News, DB Pedia [here](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset), IMDB [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### 预处理数据

对于雅虎数据，我们拼接问题和标题，问题内容和最佳答案形成一个文本去分类， 预处理的雅虎问答数据可以通过这里下载 [here](https://drive.google.com/file/d/1IoX9dp_RUHwIVA2_kJgHCWBOLHsV9V7A/view?usp=sharing). 

注意AG News和 DB Pedia，我们只利用数据内容分类，不利于标题 ， 对于IMDB数据不做预处理

我们使用 [Fairseq](https://github.com/pytorch/fairseq) 进行反向翻译用于训练集数据增强, 请参考`./data/yahoo_answers_csv/back_translate.ipynb` .

这里有2个示例, `de_1.pkl and ru_1.pkl`, in `./data/yahoo_answers_csv/` as well. 你可以直接使用他们，用于Yahoo问答，去生成你的反向翻译数据`./data/yahoo_answers_csv/back_translate.ipynb`.



### 训练模型
包含雅虎问答，使用10分类的进行训练

#### 训练 BERT baseline model
首先运行`./code/normal_train.py` 训练Bert的baseline模型，只使用于有标签的训练集
```
python ./code/normal_train.py --gpu 0,1 --n-labeled 10 --data-path ../data/yahoo_answers_csv/ --batch-size 8 --epochs 20 
```

#### 训练 TMix model
运行 `./code/train.py` 训练 TMix model， 只使用于有标签的训练集
```
python ./code/train.py --gpu 0,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/ \
--batch-size 8 --batch-size-u 1 --epochs 50 --val-iteration 20 \
--lambda-u 0 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 --separate-mix True 
```


#### 训练 MixText model
运行 `./code/train.py` 训练 MixText model，同时使用使用于有标签和无标签的训练集:
```
python ./code/train.py --gpu 0,1,2,3 --n-labeled 10 \
--data-path ./data/yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
--lrmain 0.000005 --lrlast 0.0005
```





