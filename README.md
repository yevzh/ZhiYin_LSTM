### Bert&LSTM 情感分析器
power by ZhiYinBoy

#### 运行环境
本项目利用 anaconda 虚拟环境，配置环境过程如下，请在项目目录下输入以下命令：
```
conda create -n ZhiYinBoy python=3.9
conda activate ZhiYinBoy
pip install -r requirements.txt
```

#### 测试过程
将需要测试的中英文 .xml 文件分别命名为 cn_input.xml 和 en_input.xml, 放入 Input 文件夹下，先利用以下命令进行格式转换：
```
python translib.py -t cn -i cn_input.xml -o cn_input.csv
python translib.py -t en -i en_input.xml -o en_input.csv
```

而后利用如下命令利用不同模型进行测试
```
python testlib.py -l cn -m my -t cn_input.csv
python testlib.py -l cn -m cf -t cn_input.csv
python testlib.py -l en -m my -t en_input.csv
python testlib.py -l en -m cf -t en_input.csv
```

预测得到的 .xml 文件可以在 Output 文件夹中找到

#### 训练过程
同测试过程，先进行格式转换：
```
python translib.py -t cn -i cn_input.xml -o cn_input.csv
python translib.py -t en -i en_input.xml -o en_input.csv
```
而后选择训练集加以训练
```
python train.py -t cn --train_file train.csv --test_file test.csv
python train.py -t en --train_file train.csv --test_file test.csv
```
可以选择不同的模型和方法进行训练，并调整训练参数，具体参数详见 config.py

