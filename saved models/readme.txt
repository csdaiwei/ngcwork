models文件夹下存储训练好的分类器模型。

模型使用model_generate.py生成，简述如下：
1. 使用pickle存储，字典结构，包含LR\RandomForest\Xgboost三个分类器
2. 每个传感器组生成一个模型文件，例如Geno DS传感器组，生成Geno DS.models文件 
3. 混合HHL和BY数据，采样90%，并使用EasyEnsemble重采样10次，作为训练数据
4. label reduction 采用 <=2 方式

需到目录../classification_test下运行model_generate.py脚本


predict.py读取预训练完成的模型，对输入的测试样本进行预测

程序运行方式：
usage: $ python predict.py [npzfilename]
as     $ python predict.py MB.npz

