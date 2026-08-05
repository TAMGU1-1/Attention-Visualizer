# Transformer Visualizer

문장을 입력하면 Self-Attention 계산을 단계별 히트맵으로 보여주는 Next.js 앱입니다.

$$\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

수식 한 줄을 7단계(임베딩 → Q/K/V → Raw Score → Scaling → Softmax → Value 가중합 → Output Projection)로 쪼개서, 각 단계의 실제 숫자 행렬을 표시합니다. 행을 클릭하면 그 토큰이 모든 단계에서 함께 하이라이트됩니다.

## 실행

```bash
npm install
npm run dev
```

http://localhost:3000

## 두 가지 가중치 모드

| 모드 | 임베딩 | W<sub>Q</sub>/W<sub>K</sub>/W<sub>V</sub> | 결과 |
|---|---|---|---|
| **Pretrained BERT** (기본) | bert-tiny 학습된 임베딩 | **학습된 가중치** | 항상 동일 |
| **GloVe + random** | GloVe 50d | 매번 새로 뽑는 난수 | 매번 다름 |

GloVe 모드는 "계산 과정"만 실제이고 attention 패턴에는 의미가 없습니다. 학습 전/후를 비교하는 용도로 남겨둔 것입니다.

### Pretrained BERT 모드에서 실제로 계산하는 것

`google/bert_uncased_L-2_H-128_A-2` (bert-tiny, 128d · 2 layers · 2 heads · head_dim 64)의 가중치를 그대로 사용합니다. 추론 라이브러리를 쓰지 않고 **브라우저에서 직접 순전파를 돌립니다.**

- WordPiece 토크나이즈 (`[CLS]` / `[SEP]` 포함, `visualizing` → `visual` + `##izing`)
- 학습된 **positional embedding** + segment embedding + LayerNorm
- bias 포함한 Q/K/V 투영, 지정한 **layer / head**로 슬라이스
- 모든 층을 attention과 FFN까지 전부 실행 (중간값은 선택한 층에서 캡처)

### 벡터를 단어로 되돌려 읽기

- **Q · K · V 판독** — 세 벡터가 각각 어떤 단어를 가리키는지 읽어냅니다. Q는 "내가 찾는 것", K는 "나를 찾는 것"(어휘 30,522개와 대조), V는 "내가 전하는 것"(`v·W_O`를 디코딩). `visual`의 Q는 `visualization, visualised, visualizes`를, V는 `visual, vision, visually`를 가리킵니다. layer 0에서만 동작하며 약 1초가 걸려 버튼을 눌렀을 때만 계산합니다.
- **Logit Lens** — 은닉 벡터를 MLM 헤드로 디코딩해 단어 분포로 되돌립니다. 문장에 `[MASK]`를 넣고 그 토큰을 선택하면 **모델이 빈칸을 채웁니다.**

## 모델 에셋 다시 만들기

`public/models/bert-tiny/` 는 커밋되어 있으므로 보통은 다시 만들 필요가 없습니다. 다른 체크포인트로 바꾸려면:

```bash
npm run model:build      # HF에서 safetensors 받아 public/models/ 로 추출
npm run model:verify     # 전치·fp16 변환이 원본과 일치하는지 검사
npm run model:test       # 토크나이저 + 순전파 + MLM 예측 종단 테스트
npm run model:test-lens  # 어휘 투영 / Query 검색 / logit lens 테스트
```

다른 모델을 쓰려면 (safetensors가 있는 저장소여야 합니다):

```bash
node scripts/build-model.mjs bert-base-uncased bert-base
node scripts/inspect-model.mjs bert-base-uncased   # 텐서 이름/모양 확인
```

`npm run model:test`의 MLM 예측이 가장 강한 검증입니다. 가중치 전치·LayerNorm·GELU·residual 중 하나라도 틀리면 예측이 무작위 단어로 무너집니다.

> ⚠️ bert-tiny는 2층 128차원이라 사실 암기는 못 합니다. `the capital of france is [MASK]` → `paris`가 아니라 나라 이름들이 나옵니다. 카테고리는 맞히지만 사실은 못 맞히는 것이 정상입니다.

### 에셋 구성

| 파일 | 크기 | 내용 |
|---|---|---|
| `word-embeddings.f16.bin` | 7.45 MB | 30522 × 128 임베딩 테이블 (half precision) |
| `core.f32.bin` | 1.95 MB | position/LayerNorm/Q·K·V·O/FFN/MLM 헤드 |
| `vocab.txt` | 0.22 MB | WordPiece 어휘 |

JSON이 아니라 바이너리인 이유: 임베딩만 390만 개 실수라 텍스트로는 약 30MB가 됩니다. 가중치는 선택한 모드에서 처음 필요할 때만 받습니다.

## 구조

```
src/app/page.js        UI 전체 (히트맵, 파이프라인 단계, 컨트롤)
src/lib/linalg.js      행렬/벡터 연산, softmax, LayerNorm, GELU
src/lib/wordpiece.js   WordPiece 토크나이저
src/lib/bertModel.js   에셋 로딩 + 텐서 접근
src/lib/bertPipeline.js 순전파 + 시각화용 중간값 캡처
src/lib/vocabLens.js   어휘 투영, Query→Key 검색, logit lens
scripts/               모델 추출·검증 스크립트 (빌드에는 포함되지 않음)
```

계산 과정에서는 **반올림하지 않습니다.** 층을 거치며 오차가 누적되기 때문에, 반올림은 표시 단계(`MatrixHeatmap`의 `precision`)에서만 합니다.
