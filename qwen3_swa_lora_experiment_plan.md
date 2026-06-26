# Full-AttentionモデルをSWA＋上位Full層へ変換するLoRA実験計画

## 1. 目的

既存のFull Attentionモデルを、以下のハイブリッド構成へ低コストで適応できるか検証する。

- 下位・中間層：固定幅SWA（Sliding Window Attention）
- 上位層：Full Attentionを維持
- 学習対象：主にSWA化した層のLoRA（Low-Rank Adaptation）
- LM head：凍結

主仮説は次のとおり。

> 上位に少数のFull Attention層を残せば、下位層をSWA化しても任意距離の過去位置を直接検索できる。下位層のLoRAを、凍結したFull Attention教師の最終隠れ状態へ一致させることで、長距離検索に適した局所表現へ適応できる。

この構成は、全層SWAの有限受容野問題を避けながら、KVキャッシュとAttention計算量を大幅に削減することを狙う。

---

## 2. 最初の対象モデル

第一候補：`Qwen/Qwen3-0.6B-Base`

選定理由：

- 小型で単一GPU検証がしやすい
- Decoder-only Transformer
- GQA（Grouped-Query Attention）を使用
- Q/KへのRMSNormとRoPEを含むため、現代的なAttention実装を検証できる
- Transformers実装がAttention backendを分離しており、将来の共通化に向く

初期実験ではBaseモデルを使い、チャットテンプレートや思考モードの影響を避ける。

---

## 3. 教師・生徒モデル

### 3.1 教師

- 元の事前学習済みモデル
- 全層Full Attention
- 全パラメータ凍結
- `eval()` と `torch.no_grad()` で実行
- 最終RMSNorm後の隠れ状態を教師ターゲットとして保存

### 3.2 生徒

ベース重みは教師と同一に初期化する。

初期構成：

```text
layer 0 ... L-2 : 固定幅SWA + LoRA
layer L-1       : Full Attention、凍結
final norm      : 凍結
LM head         : 凍結
embedding       : 凍結
```

Qwen3では、設定値の暗黙動作に依存せず、層種別を明示する。

```python
config.layer_types = (
    ["sliding_attention"] * (config.num_hidden_layers - num_full_top_layers)
    + ["full_attention"] * num_full_top_layers
)
config.sliding_window = window_size
```

最初は `num_full_top_layers = 1` とする。

---

## 4. なぜ上位Full層を残すか

全層SWAでは、情報伝播距離は概ね層ごとのwindow幅の総和に制約される。

一方、最終層をFull Attentionにすると、現在位置は過去の全位置にある最終層用KVを直接参照できる。

下位SWA層の役割は、過去の各位置を完全に未来へリレーし続けることではなく、上位Full層から検索可能な局所・意味表現へ変換することになる。

このため、任意距離アクセスを維持しながら、Full KVを保持する層数を大幅に減らせる。

KV要素数の概算は、系列長を `T`、SWA幅を `W`、全層数を `L`、Full上位層数を `G` とすると、

```text
全層Full       : L × T
ハイブリッド   : (L - G) × W + G × T
```

となる。

---

## 5. 学習対象

### 5.1 初期LoRA対象

SWA化した各層の以下の線形層にLoRAを付ける。

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

追加アブレーションとしてMLPにも付ける。

- `gate_proj`
- `up_proj`
- `down_proj`

### 5.2 凍結対象

初期条件では以下を凍結する。

- ベースモデルの全重み
- 上位Full Attention層
- final norm
- LM head
- token embedding

これにより、改善がLM headの再適応ではなく、下位SWA層の表現適応に由来するかを明確にする。

---

## 6. 損失関数

### 6.1 主損失：最終隠れ状態一致

教師と生徒へ同じ系列をteacher forcingで入力する。

教師の最終RMSNorm後の状態を `z_T`、生徒を `z_S` とする。

```math
L_hidden = mean_t d(stopgrad(z_T[t]), z_S[t])
```

初期の距離関数：

```math
d(a,b) = MSE(a,b) + λ_cos (1 - cosine(a,b))
```

教師が固定されているため、生徒が全表現を定数へ潰すだけでは損失を下げられない。

### 6.2 通常CE損失：アブレーション

LM headは凍結したまま、生徒の通常のCross Entropy（CE）を追加する条件も比較する。

```math
L = L_hidden + λ_CE L_CE
```

CEは情報移転の主教師ではなく、言語能力維持の正則化として扱う。

比較条件：

- `λ_CE = 0`
- 小さい `λ_CE`
- CEのみ

### 6.3 局所カットオフ損失：第二段階

最終隠れ状態一致だけで学習が不十分な場合に追加する。

各中間層についてFull/SWAのAttention出力差を測り、その損失を直前Writer層のLoRAだけへ流す。

ただし初期ベースラインには含めない。まず単純なend-to-end最終隠れ状態一致の有効性を確認する。

---

## 7. Forwardと並列化

教師と生徒はバッチ方向へ連結せず、同一バッチを逐次forwardする。

```python
with torch.no_grad():
    teacher = teacher_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
    )
    teacher_z = teacher.last_hidden_state.detach()

student = student_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=False,
    use_cache=False,
)
student_z = student.last_hidden_state

loss = hidden_loss(student_z, teacher_z)
loss.backward()
```

実際にはCausalLM wrapperの出力ではなく、ベースモデルのfinal norm後状態を直接取得する。

逐次forwardを使う理由：

- 教師側のautograd活性を保持しない
- Full AttentionとSWAで別の最適カーネルを利用できる
- バッチ内で異なるmaskを混ぜずに済む
- メモリー使用量を抑えやすい

---

## 8. アーキテクチャ非依存化

コードは次の3層に分離する。

### 8.1 `ModelAdapter`

モデル固有の構造を吸収する。

```python
class ModelAdapter(Protocol):
    def decoder_layers(self, model): ...
    def self_attention(self, layer): ...
    def final_norm(self, model): ...
    def lm_head(self, model): ...
    def set_layer_types(self, config, policy): ...
    def lora_target_modules(self) -> list[str]: ...
```

最初に実装するAdapter：

1. Qwen3
2. Llama
3. Mistral
4. Gemma系

### 8.2 `AttentionPolicy`

層ごとのAttention種別を定義する。

```python
class AttentionPolicy:
    layer_types: list[str]
    sliding_window: int
```

例：

```text
[SWA, SWA, ..., SWA, Full]
```

### 8.3 `Trainer`

以下だけを担当する。

- 教師・生徒の逐次forward
- 損失計算
- gradient accumulation
- mixed precision
- checkpoint保存
- 評価

Q/K正規化、RoPE、GQA展開、score softcapなどは、可能な限り各モデルの元実装へ任せる。

---

## 9. GQAとSWAの扱い

GQAとSWAは独立した軸であり、組み合わせ自体に構造的な問題はない。

注意事項：

- KVキャッシュは展開前の `[B, H_kv, T, D]` で保持する
- eager検証時だけKVをQueryヘッド数へ一時展開する
- 損失はQueryヘッド方向に平均し、GQA比によるスケール差を抑える
- Full教師とSWA生徒でGQA構成は変更しない

Qwen3固有のQ/K RMSNormとRoPEは、モデル本来のforward経路を通して適用する。

---

## 10. 実装フェーズ

### Phase 0：小型Toyモデル

目的：学習コードと勾配経路の検証。

- 4〜8層
- 短い系列
- eager Attention
- Full教師／下位SWA＋最終Full生徒
- LoRA勾配の有無を層別に検査

必須テスト：

- 教師パラメータの勾配がすべて `None`
- 凍結した上位Full層の勾配が `None`
- SWA層LoRAには非ゼロ勾配
- final hidden lossが減少する
- causal maskに未来漏洩がない
- SWA window外のAttentionがゼロ
- 最終Full層は全過去を参照可能

### Phase 1：Qwen3-0.6B、単一GPU

推奨初期設定：

```yaml
model: Qwen/Qwen3-0.6B-Base
sequence_length: 2048
sliding_window: 256 or 512
num_full_top_layers: 1
lora_rank: 16
lora_alpha: 32
precision: bf16
attention_backend: eager or sdpa
use_cache: false
```

最初は少数stepで過学習試験を行い、損失が確実に下がることを確認する。

### Phase 2：長系列化

- 系列長4K、8K、16Kへ拡張
- SDPAまたはFlashAttentionへ移行
- gradient checkpointingを導入
- query位置のサンプリングが必要な局所損失は、この段階で検討

### Phase 3：分散学習

順番：

1. DDP（Distributed Data Parallel）
2. 必要ならFSDP2

確認事項：

- 未使用LoRAパラメータがないか
- 教師・生徒2 forwardによるFSDP all-gather回数
- gradient accumulation中の同期
- `torch.compile`使用前後の正しさ

### Phase 4：推論評価

学習後LoRAをマージまたはロードし、推論用のハイブリッドAttention構成で評価する。

vLLMは学習には使わず、次の用途に使う。

- 長文生成
- throughput
- time-to-first-token
- decode latency
- KVキャッシュ使用量
- 同時リクエスト性能

ハイブリッド層構成がvLLMの既存backendで扱えない場合は、まずTransformersまたは専用PyTorch推論で品質評価し、その後vLLM対応を行う。

---

## 11. データ

初期段階では自然な長文データをprefix/suffixへ分割する必要はなく、通常の連続系列をteacher forcingするだけでよい。

候補データ：

- 長文Webテキスト
- 書籍・論文形式の連続文章
- コード
- 長い会話

重要なのは、系列内に長距離依存が含まれること。

学習データと別に、以下の合成タスクを評価専用で用意する。

- Passkey retrieval
- Needle-in-a-Haystack
- 過去の数値の位置指定検索
- 複数の遠距離事実の結合
- 長文中の命令保持

---

## 12. 評価指標

### 12.1 表現・品質

- 教師／生徒のfinal hidden MSE
- final hidden cosine similarity
- 凍結LM headでのperplexity差
- 通常言語タスクの品質差
- 長距離検索タスクの正解率
- 距離別性能曲線

### 12.2 システム

- Prefill latency
- Decode latency/token
- Time to First Token
- peak GPU memory
- KV cache bytes/token
- throughput
- batch size拡張性

### 12.3 学習安定性

- 層別LoRA勾配ノルム
- LoRA更新ノルム
- teacher/student表現分散
- lossの層・位置別分布
- NaN/Inf

---

## 13. 比較条件

最低限、以下を比較する。

| ID | 構成 | 学習 |
|---|---|---|
| A | 全層Full | 元モデル教師 |
| B | 下位SWA＋最終Full | 学習なし |
| C | 下位SWA＋最終Full | 下位層LoRA、hidden loss |
| D | 下位SWA＋最終Full | 下位層LoRA、hidden loss＋CE |
| E | 全層SWA | LoRA、hidden loss |
| F | 下位SWA＋上位2〜4層Full | LoRA、hidden loss |

Aは品質上限、BはSWA化の素の劣化、Cが主提案、Eは最終Full層の価値を示す比較になる。

---

## 14. 主要アブレーション

### Attention配置

- 上位Full層数：1 / 2 / 4
- Full層を最上位へ集中させるか、周期的に配置するか
- 全層SWA

### Window

- 256 / 512 / 1024 / 2048

### LoRA

- rank：8 / 16 / 32 / 64
- Attentionのみ
- Attention＋MLP
- 上位Full層LoRAあり／なし

### 損失

- hidden MSEのみ
- hidden cosineのみ
- MSE＋cosine
- CE追加
- 中間局所カットオフ損失追加

### モデル

- Qwen3
- Llama系
- Mistral系
- Gemma系

---

## 15. 成功基準

初期Go/No-Go基準：

1. 学習なしのハイブリッドモデルより、長距離検索性能が明確に改善する
2. 教師とのperplexity差が縮小する
3. 最終隠れ状態差が検証データでも縮小する
4. 全層Fullに対してKVキャッシュを大幅に削減できる
5. 上位1〜4層Fullで、全層SWAより遠距離性能が明確に高い
6. 推論時に動的削除器や不規則なtoken selectionを必要としない

研究仮説が弱いと判断する条件：

- hidden lossは下がるが長距離タスクが改善しない
- LoRAではSWA化の劣化をほとんど回復できない
- 上位Full層を増やさないと品質が戻らず、計算削減が小さくなる
- システム上、Full/SWA混在が既存カーネルで効率化できない

---

## 16. 想定リスクと対策

### 最終隠れ状態一致が過剰制約

内部状態が教師と異なっても機能的には同等な可能性がある。

対策：

- cosineとMSEの比較
- 低ランク射影後の一致
- CKAなど表現類似度の利用
- frozen headでの性能を併用して評価

### Full上位層がボトルネック

最終層は長さ `T` のKVとAttention計算を残す。

対策：

- まず実用上の削減量を測る
- 後段研究として上位Full層だけblock-sparse化する
- 重要ブロック選択は、固定SWA変換の有効性確認後に追加する

### LoRA容量不足

対策：

- rankを増やす
- MLPにもLoRAを付ける
- 上位Full層にも小ランクLoRAを許可する
- 最終的に部分的full fine-tuningと比較する

### アーキテクチャ固有差

対策：

- Q/K/Vを再実装せず、モデル本来のAttention前処理を利用する
- モデル固有処理を`ModelAdapter`へ閉じ込める
- Gemmaのscore softcapやalternating attentionは専用Adapterで扱う

---

## 17. 最初の一週間で行う最小実験

1. Qwen3-0.6Bを読み込む
2. 下位層をSWA、最終1層をFullへ設定する
3. 学習なしでperplexityと長距離検索劣化を測る
4. 下位Attention projectionへLoRAを追加する
5. Full教師とSWA生徒を逐次forwardする
6. final norm後hidden lossで数千step学習する
7. 次の3条件を比較する
   - 全層Full教師
   - ハイブリッド、学習なし
   - ハイブリッド、LoRA学習あり
8. window 256 / 512 / 1024を比較する
9. 上位Full層数1 / 2を比較する
10. hidden loss低下が長距離検索改善へ結び付くか確認する

---

## 18. 現時点の中心命題

> 既存のFull Attention Transformerについて、上位少数層のグローバルアクセスを維持しながら下位層を固定SWAへ置換し、凍結Full教師の内部表現を用いてLoRA適応することで、動的KV選択なしに長距離能力とKV効率を両立できるか検証する。
