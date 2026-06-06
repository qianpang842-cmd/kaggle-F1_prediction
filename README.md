# kaggle-F1_prediction
项目目标：预测一级方程式车手是否会在下一圈进站。

项目概述：本比赛是基于一级方程式赛车真实和模拟博弈环境设计的高阶策略预测挑战赛。

目标输出：ID及其对应的进站概率。

数据分析：
1.对训练集数据的基本分析
<img width="1596" height="186" alt="9f51794df4d9c1939dcfb6cd5b06471c" src="https://github.com/user-attachments/assets/ba3b70d1-12fd-47c2-ab38-8b0ee643e3be" />
<img width="1599" height="762" alt="7f6c8e72227553b05d16e4a95f07a0a4" src="https://github.com/user-attachments/assets/a045e2e4-67ad-47be-9da6-5e57bfa05d04" />
<img width="1650" height="819" alt="346bf4a368e7bc6e9dc482cf27459fac" src="https://github.com/user-attachments/assets/46397fb5-0b64-4e2d-8f9d-eef794a65366" />

2.对空数据的检测及处理
<img width="1605" height="495" alt="38fd3b5c63a5075f9ce2c837d85a5143" src="https://github.com/user-attachments/assets/a9ea4386-37bc-43b9-8991-2dcdea9bdf9b" />

3.进站稀疏度分析
<img width="1611" height="1071" alt="f08084a16d474a891ca94e57a4be0a10" src="https://github.com/user-attachments/assets/600b801d-3e88-4493-8d26-6c4e2435656c" />

4.分析核心属性影响力
<img width="1668" height="1146" alt="ee6f78630f079fa3a458f0e16e485dea" src="https://github.com/user-attachments/assets/0c3a5976-7777-4612-a7b4-d07e691d555c" />
图A：基于GBDT树模型属性的重要性，让训练数据集在LightGBM 或 XGBoost 模型上跑，统计每个属性在整棵树的构建过程中，被选为分裂节点的总次数或带来的信息增益。
图B：基于信息论的互信息得分，通过信息论公式计算单个特征与目标标签（是否进站）之间的不确定性消除量。
属性贡献度定量表：图A和图B的结合。

核心算法及方案：
1.外部数据动态注入与样本加权
在每折循环内部，将外部原始数据集（Original Data）拼接至当前折的训练集中，并赋予外部数据0.65的样本权重，而比赛官方数据权重保持为 1.0。

2.特征工程
由于训练集的属性并非独立，构造特征是将零散的物理观测指标转化为具备 F1 领域洞察的动态战术信号。
磨损速率（feat-WearRace）=TyreLife/(LapNumber+1)
边缘衰减状态（feat_LapTime_div_DegradAbs）=LapTime（s）/(Cumulative_Degradation+10**(-6))
衰减空间斜率（feat_Degradation_Slope）=Cumulative_Degradation/（TyreLife+10**（-6））
翻转窗口风险（feat_PitWindow_Risk）=RaceProgress*(21-Position)

3.加权平滑内折目标编码
双层嵌套内折（Inner OOF），编码用子折数据计算、杜绝标签回流。
加权平滑TE公式，引入样本权重和全局均值平滑，小样本自动向全局均值收敛。

4.异构模型混合与集成策略
三模型加权融合：LightGBM(0.4)+XGBoost(0.35)+CatBoost(0.25)

最终成绩
<img width="1878" height="129" alt="148e7130dccd96985395e87f0508b4e0" src="https://github.com/user-attachments/assets/cdfd4525-0d16-4234-9168-8efc49c0369d" />
765/3023，大约25%
