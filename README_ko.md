## MerFT 사용 가이드 (Korean)

### 설치와 실행
- 의존성 설치: `pip install -r raft/requirements.txt`

### 데이터셋 JSON 생성기: `raft/meme_data/scripts/generate_meme_dataset_json.py`
- **--images_dir (필수)**: 밈 이미지 디렉터리. `--recursive` 사용 시 하위 폴더 전체 스캔 후 하나의 JSON으로 합칩니다.
- **--docs_dir (필수)**: `.txt` 문서 디렉터리(1행 링크, 2행 이후 본문). 파일명(stem)이 이미지 폴더 키워드(숫자 프리픽스/ 플랫폼 서픽스 제거 후)와 일치해야 매칭됩니다.
- **--output (필수)**: 출력 JSON 경로. 각 항목: `[title, keyword, image_path, doc_link, doc_text]`.
- **--recursive**: `--images_dir` 하위 폴더들을 재귀적으로 스캔.

주의: 키워드 정규화는 `10_`, `_meme_pinterest`, `_meme_reddit` 등을 제거합니다.

### 비라벨 정리 스크립트: `raft/meme_data/scripts/organize_unlabeled.py`
- **--labeled_docs_root (필수)**: 라벨된 토픽 문서 루트(예: `docs/environment`, `docs/politics`). 토픽별 멀티모달 중심(centroid)을 생성합니다.
- **--labeled_memes_root (선택)**: 라벨된 토픽 이미지 루트. 제공 시 멀티모달 중심 품질 향상.
- **--unlabeled_docs_root (필수)**: 비라벨 문서 루트(하위에 그룹 폴더 형태 권장: `docs/unlabeled/<group>/*.txt`).
- **--unlabeled_memes_root (선택)**: 비라벨 이미지 루트(문서와 동일 그룹명 폴더 권장: `memes/unlabeled/<group>/*.{jpg,png}`).
- **--output_docs_root (선택)**: 복사 대상 문서 루트(미지정 시 `--labeled_docs_root`).
- **--output_memes_root (선택)**: 복사 대상 이미지 루트(미지정 시 `--labeled_memes_root`).
- **--embedder_model (기본: clip-ViT-B-32)**: 멀티모달 임베딩 모델.
- **--assign_method [centroid|kmeans] (기본: centroid)**: 비라벨 그룹의 토픽 할당 방식. `centroid`는 라벨 토픽 중심에 최근접 할당. `kmeans`는 비라벨 군집으로 보정.
- **--k (기본: 0)**: `kmeans` 시 클러스터 수(0이면 자동).
- **--mark_done**: 복사 후 처리된 비라벨 폴더를 `[end]_` 프리픽스로 리네임하여 다음 실행에서 스킵.
- **--skip_done**: `[end]_`로 시작하는 비라벨 폴더는 스킵.

동작 요약:
- 라벨 데이터로 토픽 중심을 만들고(준지도), 비라벨 그룹을 최근접 토픽으로 할당해 문서/이미지를 해당 토픽 폴더로 복사합니다. 원본은 `[end]_`로 마킹 가능.

### QA 생성기: `raft/meme_qa_generator.py`
필수 I/O
- **--input (필수)**: 데이터셋 JSON 경로(상기 생성기 출력).
- **--documents_root (필수)**: 카테고리별 문서 루트.
- **--output (필수)**: 결과 데이터셋 경로(방해문서 수별 파일도 함께 생성).

OpenAI 및 실행
- **--openai_key (필수)**: OpenAI API 키.
- **--model (기본: gpt-4-mini)**, **--workers (기본: 4)**.

방해문서 선정 전략
- **--distractor_strategy [random|similarity|clustering] (기본: random)**
  - random: 무작위 샘플링(오라클과 앞 100자 동일 문서는 제외).
  - similarity: 멀티모달 임베딩 기반으로 오라클과 유사한 상위 k 문서 선택.
  - clustering: 오라클+풀을 KMeans로 군집화, 동일 클러스터 내 하드 네거티브 선별.

준지도/멀티모달 리소스
- **--unlabeled_docs_root (선택)**: 비라벨 문서를 풀에 포함. 파일/폴더명의 정규화된 스템을 기준으로 이미지 페어링.
- **--unlabeled_memes_root (선택)**: 비라벨 이미지 포함.
- **--embedder_model (기본: clip-ViT-B-32)**: 멀티모달 임베딩 모델.

클러스터링 제어
- **--cluster_k (기본: 20)**: `clustering` 전략의 KMeans 클러스터 수(풀 크기에 맞춰 자동 보정).
- **--cluster_output (선택)**: `cluster_{id}/docs`, `cluster_{id}/images`로 클러스터 물질화(복사 기반).
- **--cluster_min_size (기본: 10)**: 물질화 시 최소 클러스터 크기 필터.

기타
- 방해문서 개수는 `[0,1,2,3,4,5]` 전부 생성. `k=5`에서는 오라클 문서를 제외합니다.
- 풀 크기가 부족하면 더미 문서로 보충(오라클과 앞부분 동일한 경우 제외).

## 왜 MerFT인가? 어떤 문제를 푸나?

MerFT는 검색 기반 멀티모달 환경에서 노이즈 문서(무관/혼동 유발)가 섞여도 견고하게 밈을 해석하도록 설계되었습니다.

- 이미지·캡션·문서를 결합해 문화 맥락까지 파악
- 문헌 인용 기반 CoT로 근거가 드러나는 추론
- 준지도 클러스터링으로 비라벨 데이터 흡수 및 near-miss 하드 네거티브 선별

활용 대상:
- 사회 갈등·풍자 분석팀, 문화 서사 추적 연구자
- 노이즈에 강한 RAG 기반 VLM을 구축하려는 실무자
- 설명 가능한 문서-근거형 QA가 필요한 응용

## 전체 워크플로우(원샷)

1) 입력 JSON 생성(재귀 스캔)
```bash
python raft/meme_data/scripts/generate_meme_dataset_json.py \
  --images_dir raft/meme_data/memes \
  --docs_dir raft/meme_data/docs/environment \
  --output raft/meme_data/scripts/environment.json \
  --recursive
```

2) 비라벨 그룹 토픽 정리(복사 + 마킹)
```bash
python raft/meme_data/scripts/organize_unlabeled.py \
  --labeled_docs_root raft/meme_data/docs \
  --labeled_memes_root raft/meme_data/memes \
  --unlabeled_docs_root raft/meme_data/docs/unlabeled \
  --unlabeled_memes_root raft/meme_data/memes/unlabeled \
  --embedder_model clip-ViT-B-32 \
  --assign_method centroid \
  --mark_done --skip_done
```

3) 클러스터링 기반 하드 네거티브로 QA 생성
```bash
python raft/meme_qa_generator.py \
  --input raft/meme_data/scripts/environment.json \
  --documents_root raft/meme_data/docs/environment \
  --unlabeled_docs_root raft/meme_data/docs/unlabeled \
  --unlabeled_memes_root raft/meme_data/memes/unlabeled \
  --embedder_model clip-ViT-B-32 \
  --distractor_strategy clustering \
  --cluster_k 32 \
  --cluster_output clustered_output \
  --cluster_min_size 10 \
  --output output.json \
  --openai_key $OPENAI_API_KEY \
  --model gpt-4o-mini \
  --workers 2
```

변형: `--distractor_strategy similarity`(유사도 기반), `random`(기준선)도 선택 가능.

## 논문과의 매칭

- 멀티모달 입력 모드(Base/Caption/Both)
- 방해문서 수 가변(k=0..5, k=5는 오라클 제외)
- 준지도 멀티모달 클러스터링으로 비라벨 흡수 및 동일 클러스터 하드 네거티브 선별
- 근거 표기형 CoT 유도와 최종 정답 태깅

결과적으로 노이즈가 많은 검색 환경에서도 해석 정확도와 설명 가능성을 동시에 달성합니다.


