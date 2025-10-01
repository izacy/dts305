# nlp cw

**DTS305TC cw1**

## 项目结构

```
├── main_system.py              
├── algorithm1_lsi.py           
├── algorithm2_language_model.py 
├── evaluation.py               
├── evaluation_runner.py             
├── dataset/                    
├── result
│   ├── precision.csv               
│   ├── recall.csv                  
│   ├── nDCG.csv                    
│   └── relevance_judgments.csv     
├── README.md                   
├── requirements.txt            
└── 25-26 DTS305TC_CW1_F.docx   
```

## 安装配置

### 环境要求
- Python 3.10 +

### 第一步：克隆仓库
```bash
git clone https://github.com/izacy/dts305.git
cd dts305
```

### 第二步：安装依赖
```bash
pip install -r requirements.txt
```


## 使用方法

### 运行主系统
```bash
python main_system.py
```

系统将会：
1. 从 `dataset/` 加载并预处理所有文档
2. 构建LSI和语言模型索引
3. 启动交互式查询界面

### 交互式查询界面
```
> 英国运动员跨栏
```

**输出示例：**
```
查询: 英国运动员跨栏
----------------------------------------
LSI算法结果:
1. 001.txt (得分: 0.847)
2. 002.txt (得分: 0.692)
...

语言模型算法结果:
1. 001.txt (得分: -12.543)
2. 003.txt (得分: -14.872)
...

加权融合结果 (α=0.6):
1. 001.txt (融合得分: 0.923)
2. 002.txt (融合得分: 0.754)
...
```

### 运行评估模块
```bash
python evaluation_runner.py
```

### 支持的查询命令
- **普通查询**：输入任何文本进行检索
- **退出系统**：输入 `quit`、`exit` 或 `q`

## 算法详解

### 算法1：潜在语义索引（LSI）
- **原理**：使用奇异值分解（SVD）发现文档中的潜在语义关系
- **优势**：处理同义词问题，基于概念而非关键词检索
- **技术**：TF-IDF + SVD降维 + 余弦相似度

### 算法2：查询似然语言模型
- **原理**：计算查询从文档生成的概率
- **优势**：统计严谨，自然的概率排序
- **技术**：Jelinek-Mercer平滑 + 最大似然估计

### 融合策略
- **加权融合**：α × LSI得分 + (1-α) × 语言模型得分
- **默认权重**：α = 0.6（LSI占60%，语言模型占40%）

## 评估方法

### 自动评估系统（减小人为导致误差）
1. **查询生成**：从文档内容提取关键短语作为查询
2. **相关性判断**：基于内容分析自动创建相关性评分
3. **多样性保证**：选择不同类型和长度的查询

### 评估指标
- **Precision@5**：前5个结果中相关文档的比例
- **Recall@5**：前5个结果找到的相关文档占总相关文档的比例  
- **nDCG@5**：标准化折损累积增益，考虑排序质量

### 输出文件
- `precision.csv`：各算法精确率对比
- `recall.csv`：各算法召回率对比
- `nDCG.csv`：各算法nDCG得分对比
- `relevance_judgments.csv`：自动生成的相关性判断

## 系统架构

### 核心功能

1. **主函数**：控制整个信息检索系统的启动和流程
2. **用户输入函数**：允许用户在控制台连续输入检索文本
3. **数据库输入函数**：从文档文件夹读取本地文本库
4. **文本预处理函数**：预处理读取的文档
5. **信息检索算法1**：使用LSI算法检索并提供前5个候选结果
6. **信息检索算法2**：使用语言模型算法检索并提供前5个候选结果
7. **输出函数**：输出两种检索算法的结果
8. **融合输出函数**：使用加权合并方法融合搜索结果并打印新排序


### 示例查询
```bash
# 启动系统
python main_system.py

# 输入查询
> british hurdles championship
> european athletics competition  
> medal winning performance
> sports record breaking
```

### 评估运行
```bash
# 运行完整评估
python evaluation_runner.py

# 查看结果
ls results/
precision.csv  recall.csv  nDCG.csv  relevance_judgments.csv
```


## 提交指南

1. 创建功能分支 (`git checkout -b feature/新功能`)
2. 提交更改 (`git commit -am '添加新功能'`)
3. 推送到分支 (`git push origin feature/新功能`)
4. 创建Pull Request
