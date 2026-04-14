# dawonv5 Summary Examples

## Finance Example 1: 분기보고서 기본정보

**Raw anchor text (일부):**
```
证券代码：600310  证券简称：广西能源
本公司董事会及全体董事保证本公告内容不存在任何虚假记载...
|项目|本报告期|上年同期|本报告期比上年同期增减变动幅度(%)|
|营业收入|969,660,264.44|5,928,790,478.64|-83.64|
```

**Extracted structured metadata:**
```json
{
  "anchor_owner_type": "company",
  "anchor_owner_name": "广西能源股份有限公司",
  "anchor_content_type": "table",
  "anchor_semantic_role": "statistic",
  "anchor_time_scope": "2024Q1",
  "anchor_unit_hints": ["单位：元", "元", "%"],
  "v5_confidence": 0.86
}
```

**Rendered summary:**
> 广西能源股份有限公司 > '基本信息'의 통계 표. 广西能源股份有限公司2024年第一季度报告，报告未经审计。 [2024Q1 | 单位：元/元 | DOC1_A2 -> DOC1_A1 : cites_previous]

---

## Finance Example 2: 재무제표 상세

**Raw anchor text (일부):**
```
(一)主要会计数据和财务指标
|营业收入|969,660,264.44|5,928,790,478.64|-83.64|
|归属于上市公司股东的净利润|-57,302,152.03|33,408,840.44|-271.52|
```

**Extracted structured metadata:**
```json
{
  "anchor_owner_type": "company",
  "anchor_owner_name": "广西能源股份有限公司",
  "anchor_content_type": "table",
  "anchor_semantic_role": "statistic",
  "anchor_time_scope": "2024Q1",
  "anchor_unit_hints": ["万元", "%"],
  "citation_edges_out": [{"target": "DOC1_A1", "type": "cites_previous"}],
  "v5_confidence": 0.86
}
```

**Rendered summary:**
> 广西能源股份有限公司 > '주요회계데이터'의 통계 표. 报告期营业收入969,660,264.44元，同比下降83.64%。 [2024Q1 | 万元/% | DOC1_A2 -> DOC1_A1 : cites_previous]

---

## Legal Example 1: 법원 판단 (holding)

**Raw anchor text (일부):**
```
本院认为，被告在签订合同时未履行告知义务，违反了诚实信用原则。
根据《合同法》第五十四条之规定，原告请求撤销合同，于法有据。
```

**Extracted structured metadata:**
```json
{
  "anchor_owner_type": "court_case",
  "anchor_owner_name": "判决文书1",
  "anchor_content_type": "paragraph",
  "anchor_semantic_role": "holding",
  "v5_confidence": 0.58
}
```

**Rendered summary:**
> 判决文书1 > '법원판단'의 법원 판단 문단. 被告在签订合同时未履行告知义务，违反诚实信用原则，原告请求撤销合同于法有据。

---

## Legal Example 2: 사실관계 (case_fact)

**Raw anchor text (일부):**
```
抗诉机关为河北省人民检察院。申诉人为王某庆，因诉永清县韩村镇人民政府
及第三人永清县韩村镇西庄窠村民委员会拆迁安置协议一案...
```

**Extracted structured metadata:**
```json
{
  "anchor_owner_type": "court_case",
  "anchor_owner_name": "判决文书1",
  "anchor_content_type": "paragraph",
  "anchor_semantic_role": "legal_argument",
  "v5_confidence": 0.44
}
```

**Rendered summary:**
> 判决文书1 > '안건정보'의 법적 논증 문단. 抗诉机关为河北省人民检察院，申诉人王某庆因拆迁安置协议纠纷提起诉讼。

---

## Paper Example 1: Methodology

**Raw anchor text (일부):**
```
We propose a multi-scale attention mechanism that combines local and global
context features. The model architecture consists of three components:
(1) patch embedding, (2) cross-attention module, (3) classification head.
```

**Extracted structured metadata:**
```json
{
  "anchor_owner_type": "paper",
  "anchor_owner_name": "논문 제목",
  "anchor_content_type": "paragraph",
  "anchor_semantic_role": "methodology",
  "v5_confidence": 0.58
}
```

**Rendered summary:**
> 논문 > 'Methodology'의 방법론 문단. Multi-scale attention mechanism combining local/global context; three components: patch embedding, cross-attention, classification head.

---

## Paper Example 2: Results Table

**Raw anchor text (일부):**
```
Table 3: Ablation study on CIFAR-100.
|Model|Accuracy|F1|Params|
|Baseline|78.2|76.1|12M|
|Full Model|85.4|84.2|20M|
```

**Extracted structured metadata:**
```json
{
  "anchor_owner_type": "paper",
  "anchor_owner_name": "논문 제목",
  "anchor_content_type": "table",
  "anchor_semantic_role": "statistic",
  "citation_edges_in": [{"source": "DOC1_A5", "type": "references_table"}],
  "v5_confidence": 0.58
}
```

**Rendered summary:**
> 논문 > 'Experiments'의 통계 표. Ablation study on CIFAR-100: full model 85.4% accuracy. [% | DOC1_A5 -> DOC1_A6 : references_table]
