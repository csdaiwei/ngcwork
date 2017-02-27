test_concatenate.py

对于一个部件(主轴\齿轮箱\发电机)的每个传感器组，拼接其所有的信号通道的特征，做分类测试。
以发电机(geno)为例，发电机包含驱动端(geno ds)和非驱动端(geno nds)两个传感器组，程序会分别处理这两个传感器组。
驱动端包含4个信号通道，分别是Geno DS VEL(low class), Geno DS VEL(high class), Geno DS ENV3(low class), Geno DS ENV3(high class), 则拼接这4个信号通道的特征，生成特征矩阵X，再使用发电机驱动端的标记作为Y，做分类测试。非驱动端同样方式处理。
分类模型使用sklearn库提供的LogisticRegression, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
对BY或HHL的每个部件单独做分类测试。

执行脚本顶端可调整的参数有：
LR = 2              # label reduction
REPEAT = 10         # repeat nums
TN_THRES = 0.9      # true negative thresholds
TRAIN_FACTOR = 0.7  # training set factor

程序运行方式：
usage: $ python test_concatenate.py [picklefilename]
as     $ python test_concatenate.py hhl_mb_6498.pickle



test_xgboost.py

使用xgboost做与test_concatenate.py相同的分类测试

程序运行方式：
usage: $ python test_xgboost.py [picklefilename]
as     $ python test_xgboost.py hhl_mb_6498.pickle


test_merge.py

合并BY和HHL的数据，再做与test_concatenate.py相同的分类测试

程序运行方式：
usage: $ python test_merge.py [mb/gear/geno]
as     $ python test_merge.py mb



test_cross.py

以HHL数据做训练，BY数据做测试，特征处理方式同test_concatenate.py

程序运行方式：
usage: $ python test_cross.py [mb/gear/geno]
as     $ python test_cross.py mb