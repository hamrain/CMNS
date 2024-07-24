from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
model = BertModel.from_pretrained('./bert-base-uncased')

# 定义两句话
sentence1 = "A plane flies near the sunset."

#A Helicopter rapidly crosses sky.
#A plane is far from sunrise.
#Seagull flies over beach.
#Boat cruising by the seaside.

sentence2 = "Seagull flies over beach"
# kites dance in the sky.
# boat sailing under the sunset
# 对句子进行编码
inputs1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True)
inputs2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True)

# 通过BERT模型获取句子嵌入
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# 提取[CLS]标记的隐藏状态作为句子嵌入
sentence1_embedding = outputs1.last_hidden_state[:, 0, :]
sentence2_embedding = outputs2.last_hidden_state[:, 0, :]

# 计算余弦相似度
cosine_similarity = F.cosine_similarity(sentence1_embedding, sentence2_embedding)
similarity_score = cosine_similarity.item()

print(f"The similarity score between the two sentences is: {similarity_score}")