import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
from rank_bm25 import BM25Okapi

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn: Optional[str] = 'klue/bert-base',
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface tokenize_fn
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        self.tokenize_fn = tokenize_fn
        self.data_path = data_path # wikipedia doc
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in self.wiki.values()])
        )
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # embedding을 위한 str으로 만들어 질 2차원 list
        self.contexts_for_embb = []


        print(f'Length of contexts for embedding : {len(self.contexts_for_embb)}')
        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        
    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"BM25_embedding_bert_doc.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            for context in self.contexts:
                # print(f"v['text'] : {v['text']}")
                # print(f"tokenized v : {tokenize_fn(context)}")
                self.contexts_for_embb.append(self.tokenize_fn(context))
            
            # print(f'self.context_for_embb : {self.contexts_for_embb}')
            self.p_embedding = BM25Okapi(self.contexts_for_embb)
            # print(f'[shape of p_embedding] : {self.p_embedding.shape}')
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        print('in retrieve')
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        
        # query_or_dataset = pd.read_csv('/opt/ml/input/data/train_question.csv')[:100]
        # print(f'query_or_dataset_in_retrieve : {query_or_dataset}')
        print(f'query_or_dataset_in_retrieve : {type(query_or_dataset)}')
        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, Dataset):
            print('queries are DataFrame')

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk,
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question" : example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context" : " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]] # pid : int
                    ),
                    # "score" : doc_scores
                    }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            # print(f'cqas : {cqas}')
            return cqas

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        print('[into the get_relevant_doc_bulk]')
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        doc_scores = []
        doc_indices = []
        for query in tqdm(queries, desc='computing query'):
            tokenized_query = self.tokenize_fn(query)
            result = self.p_embedding.get_scores(tokenized_query)
            # print(f'similarity result : {result}')
            # if not isinstance(result, np.ndarray):
            #     result = result.toarray()
                
            sorted_result = np.argsort(result)[::-1]
            k_indices = sorted_result.tolist()[:k]
            score = result[sorted_result][:k]#??

            # print(f'[sorted_result] : {sorted_result}')    
            # print(f'[k_indices] : {k_indices}')
            # print(f'[score] : {score}')

            doc_indices.append(k_indices) 
            doc_scores.append(score)

        return doc_scores, doc_indices

# if __name__ == '__main__':
    # data_path = '/opt/ml/input/data'
    # dataset_name = "/opt/ml/input/data/train_dataset"
    # model_name_or_path = 'klue/bert-base'
    # context_path = 'wikipedia_documents.json'
    # use_faiss = False
    # k = 20

    # # Test sparse
    # org_dataset = load_from_disk(dataset_name)
    # # print(org_dataset) 있어
    # full_ds = concatenate_datasets(
    #     [
    #         org_dataset["train"].flatten_indices(),
    #         org_dataset["validation"].flatten_indices(),
    #     ]
    # )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트

    # # 실험을 위한 query 수 조절
    # full_ds = full_ds.select(range(20))

    # print("*" * 40, "query dataset", "*" * 40)
    # print(f'[full_ds] : {full_ds}')
    # print(f'[type of full_ds] : {type(full_ds)}')

    # # white tokenizer
    # def white_tokenizer(a : str) -> List:
    #     return list(a.spit(" "))

    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False,)


    # def white_tokenizer(a : str) -> List:
    #     return list(a.split(" "))

    # retriever = SparseRetrieval(
    #     tokenize_fn=tokenizer.tokenize,
    #     # tokenize_fn=white_tokenizer,
    #     data_path=data_path,
    #     context_path=context_path,
    # )

    # retriever.get_sparse_embedding()

    # def check_correct(a, b):
    #     if b in a :
    #         return True
    #     else :
    #         return False

    # with timer("bulk query by exhaustive search"):
    #     df = retriever.retrieve(full_ds, topk = k)
    #     df["correct"] = df.apply(lambda x : check_correct(str(x['context']), str(x['original_context'])))
    #     print(
    #         "correct retrieval result by exhaustive search",
    #         df["correct"].sum() / len(df),
    #     )
        
    #     with open('/opt/ml/input/data/train_val_q_on_document', "wb" ) as file:
    #         pickle.dump(df, file)