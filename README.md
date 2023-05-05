# level2_mrc_nlp-14  

# 팀 소개  

### 

|김광연|김민준|김병준|김상혁|서재명|
| :-: | :-: | :-: | :-: | :-: |
|![광연님](https://user-images.githubusercontent.com/59431433/217448461-bb7a37d4-f5d4-418b-a1b9-583b561b5733.png)|![민준님](https://user-images.githubusercontent.com/59431433/217448432-a3d093c4-0145-4846-a775-00650198fc2f.png)|![병준님](https://user-images.githubusercontent.com/59431433/217448424-11666f05-dda6-406d-95e8-47b3bab7c2f6.png)|![상혁2](https://user-images.githubusercontent.com/59431433/217448849-758c8e25-87db-4902-ab06-0aa8c359500c.png)|![재명님](https://user-images.githubusercontent.com/59431433/217448416-b2ba2070-6cfb-4829-a3bd-861f526cb74a.png)|

## 프로젝트 주제

-  Open-Domain Question Answering

## 프로젝트 개요

- 질문에 관련된 문서를 찾아주는 Retriever 모델 만들기
- 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 Reader 모델 만들기

## 데이터 셋
- trin data
  - train: 3,952
  - validation: 240
- test data: 7,765
  - validatoin: 600

### 데이터 예시
- question id: 질문 id
- question: 질문
- answers: 답변에 대한 정보
  - answer_start: 답변의 시작 위치
  - text: 답변의 텍스트
- context: 답변이 포함된 문서
- title: 문서의 제목
- document id: 문서 id


## 평가방법
### Exact Match (EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어집니다. 즉 모든 질문은 0점 아니면 1점으로 처리됩니다. 

### F1 Score: EM과 다르게 부분 점수를 제공합니다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 받을 수 있습니다.
