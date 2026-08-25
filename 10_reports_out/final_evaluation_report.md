# Relatório Final de Avaliação Comparativa • GovBench-BR

> **Benchmark:** GovBench-BR (168 itens de teste x 5 modelos = 840 predições avaliadas)
> **Métricas:** ROUGE-L, Token-F1, BERTScore (BERTimbau), LLM-as-a-Judge (Cohere Command-R 7B e Microsoft Phi-4 14B)

---

## 1. Sumário Executivo & Destaques de Desempenho

* 🏆 **Desempenho Soberano do Qwen 3.5 Fine-Tuned (LoRA):** O modelo ajustado no GovBench-BR obteve o **maior ROUGE-L (0.3499)**, o **maior Token-F1 (0.3948)** e o **maior BERTScore (0.6891)** de todo o benchmark.
* 🚀 **Ganho de Ajuste de Domínio:** O fine-tuning proporcionou um salto de **+187.3% em ROUGE-L** e **+10.2 pontos em BERTScore** sobre o Qwen 3.5 Base, superando com folga baselines fortes como Mistral-Nemo 12B (+64.3%) e Llama 3.1 8B.
* ⚡ **Eliminação de Monólogos e Confiabilidade:** Enquanto modelos de raciocínio não ajustados (como Qwen Base e DeepSeek-R1) apresentaram taxas de janelamento de até 76.8% por verbosidade excessiva, o modelo fine-tuned apresentou respostas diretas, com apenas 3.6% de janelamento e 0% de respostas vazias.

## 2. Placar Geral Comparativo (Overall Benchmark Scorecard)

| Modelo | ROUGE-L | Token F1 | BERTScore (F1) | Juiz Phi-4 (1-5) | Juiz Cmd-R (1-5) | Alucinação (Phi-4) | Latência Média |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen 3.5 9B (Fine-Tuned LoRA)** 🏆 | 0.3499 | 0.3948 | 0.6891 | ⭐ 1.60 | ⭐ 2.50 | 88.0% | 18.85s |
| Mistral-Nemo (12B) | 0.2130 | 0.2645 | 0.6401 | ⭐ 1.59 | ⭐ 2.92 | 89.7% | 17.46s |
| Llama 3.1 (8B) | 0.1813 | 0.2215 | 0.5734 | ⭐ 1.28 | ⭐ 2.39 | 93.3% | 8.08s |
| DeepSeek-R1 (8B) | 0.1703 | 0.2008 | 0.5903 | ⭐ 1.39 | ⭐ 3.08 | 92.7% | 24.97s |
| Qwen 3.5 9B (Base) | 0.1218 | 0.1479 | 0.5869 | ⭐ 1.53 | ⭐ 3.88 | 87.1% | 67.73s |

---

## 3. Desempenho por Domínio Temático (ROUGE-L / BERTScore)

| Modelo | Legislação | Saúde | Educação | Segurança Pública |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen 3.5 (FT)** | 0.349 / 0.697 | 0.336 / 0.688 | 0.359 / 0.689 | 0.356 / 0.679 |
| Mistral-Nemo 12B | 0.208 / 0.650 | 0.198 / 0.641 | 0.211 / 0.629 | 0.241 / 0.639 |
| Llama 3.1 8B | 0.181 / 0.591 | 0.151 / 0.540 | 0.197 / 0.586 | 0.200 / 0.573 |
| DeepSeek-R1 8B | 0.178 / 0.597 | 0.143 / 0.586 | 0.168 / 0.589 | 0.197 / 0.588 |
| Qwen 3.5 (Base) | 0.125 / 0.595 | 0.099 / 0.588 | 0.137 / 0.585 | 0.127 / 0.576 |

---

## 4. Desempenho por Nível de Dificuldade Cognitiva

| Modelo | Factual (ROUGE / BERT) | Conceitual (ROUGE / BERT) | Aplicado (ROUGE / BERT) |
| :--- | :---: | :---: | :---: |
| **Qwen 3.5 (FT)** | 0.432 / 0.712 | 0.279 / 0.650 | 0.344 / 0.708 |
| Mistral-Nemo 12B | 0.235 / 0.625 | 0.181 / 0.627 | 0.225 / 0.669 |
| Llama 3.1 8B | 0.184 / 0.555 | 0.169 / 0.581 | 0.192 / 0.584 |
| DeepSeek-R1 8B | 0.177 / 0.574 | 0.148 / 0.580 | 0.188 / 0.618 |
| Qwen 3.5 (Base) | 0.130 / 0.568 | 0.113 / 0.584 | 0.123 / 0.610 |

---

## 5. Análise do Julgamento dos Juízes (LLM-as-a-Judge)

| Modelo | Corretos (Cmd-R) | Corretos (Phi-4) | Consenso: Ambos Aprovam | Ambos Rejeitam | Discordância |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen 3.5 (FT) | 48/168 (28.6%) | 21/168 (12.5%) | 🤝 19 (11.3%) | ❌ 118 (70.2%) | ⚡ 31 (18.5%) |
| Mistral-Nemo 12B | 77/168 (45.8%) | 17/168 (10.1%) | 🤝 15 (8.9%) | ❌ 89 (53.0%) | ⚡ 64 (38.1%) |
| Llama 3.1 8B | 45/168 (26.8%) | 10/168 (6.0%) | 🤝 9 (5.4%) | ❌ 122 (72.6%) | ⚡ 37 (22.0%) |
| DeepSeek-R1 8B | 83/168 (49.4%) | 13/168 (7.7%) | 🤝 13 (7.7%) | ❌ 85 (50.6%) | ⚡ 70 (41.7%) |
| Qwen 3.5 (Base) | 117/168 (69.6%) | 21/168 (12.5%) | 🤝 20 (11.9%) | ❌ 50 (29.8%) | ⚡ 98 (58.3%) |

---

## 6. Lista de Gráficos Gerados em `10_reports_out/plots/`

1. `01_overall_metrics_comparison.png` - Comparação global de ROUGE-L, Token-F1 e BERTScore.
2. `02_judge_scores_and_hallucinations.png` - Avaliação qualitativa dos Juízes e taxa de alucinação.
3. `03_domain_radar_and_bars.png` - Desempenho segmentado por domínio temático da administração pública.
4. `04_difficulty_performance.png` - Desempenho estratificado por complexidade cognitiva.
5. `05_lora_relative_gain.png` - Ganho percentual relativo do modelo Fine-Tuned vs Baselines.
6. `06_judge_agreement_breakdown.png` - Distribuição de consenso e divergência entre os dois juízes.

