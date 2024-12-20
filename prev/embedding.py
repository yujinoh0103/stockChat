from sentence_transformers import SentenceTransformer
import json

# 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 뉴스 JSON 파일 불러오기
with open('cleaned_data.json', 'r', encoding='utf-8') as file:
    news_data = json.load(file)

# 제목과 본문을 따로 임베딩
title_embeddings = []
content_embeddings = []

for article in news_data:
    title = article['title']
    content = article['content']
    
    # 제목과 본문을 따로 임베딩
    title_emb = model.encode(title)
    content_emb = model.encode(content)
    
    title_embeddings.append(title_emb)
    content_embeddings.append(content_emb)

# 임베딩을 JSON 형태로 저장
output_data = {
    "title_embeddings": [emb.tolist() for emb in title_embeddings],
    "content_embeddings": [emb.tolist() for emb in content_embeddings]
}

with open('news_embeddings_separated.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
