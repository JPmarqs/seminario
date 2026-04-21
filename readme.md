# CTGAN vs Faker: Comparativo de Geração de Dados Sintéticos

**Disciplina:** Linguagem de Programação
**Data:** 2026

---

## 📋 Objetivo do Trabalho

Demonstrar empiricamente as **vantagens do CTGAN** (IA generativa) sobre o **Faker** (abordagem tradicional baseada em regras) para geração de dados sintéticos.

---

## 🎯 Principais Conclusões

| Aspecto | Faker | CTGAN |
|---------|-------|-------|
| **Preservação de correlações** | ❌ Não preserva | ✓ Preserva |
| **Mecanismo** | Regras fixas + aleatoriedade | Rede neural generativa |
| **Tempo de geração** | ~0.01s (instantâneo) | ~60-120s (treino) + <1s (geração) |
| **Utilidade estatística** | Baixa | Alta |
| **Privacidade** | Máxima | Alta |
| **Indicado para** | Prototipação, testes simples | Pesquisa, ML, conformidade LGPD |

---

## 🚀 Instalação

### Opção 1: Instalação Completa (Recomendada)

```bash
pip install sdv faker pandas numpy matplotlib seaborn scipy
```

### Opção 2: Instalação Mínima (Apenas Faker)

```bash
pip install faker pandas numpy matplotlib seaborn scipy
```

> **Nota:** Sem o SDV, o script usará uma simulação para demonstrar os conceitos.

---

## ▶️ Execução

```bash
python seminario.py
```

**Tempo estimado:** 2-3 minutos (com CTGAN)  
**Saída:** 
- Tabelas comparativas no terminal
- Gráfico: `comparativo_ctgan_faker.png`
- Relatório: `relatorio_stack_e_*.json`

---

## 📊 Métricas Avaliadas

### 1. Correlação Salário × Score de Crédito
- **Dataset Real:** ~0.70 (correlação positiva forte)
- **Faker:** ~0.05 (correlação destruída)
- **CTGAN:** ~0.65-0.72 (correlação preservada)

### 2. Kolmogorov-Smirnov (Distribuições)
Mede quão similares são as distribuições marginais de cada coluna.

### 3. Tempo de Processamento
- Faker: instantâneo
- CTGAN: requer treinamento prévio

---

## 🧠 Fundamentação Teórica

### Por que o Faker falha em preservar correlações?

O Faker gera **cada coluna independentemente**, sem considerar relações entre variáveis. No mundo real:

- Pessoas com maior escolaridade tendem a ter maior salário
- Pessoas com maior salário tendem a ter melhor score de crédito
- **Conclusão:** Escolaridade e score devem estar correlacionados

O Faker não captura essa relação → **dados pouco realistas**.

### Como o CTGAN preserva correlações?

O CTGAN usa **Redes Generativas Adversariais (GANs)**:

```
┌─────────────┐         ┌─────────────┐
│  Gerador    │ ──────→ │ Dados Falsos │
│  (cria)     │         │              │
└─────────────┘         └──────┬──────┘
                               │
┌─────────────┐         ┌──────▼──────┐
│  Dados Reais│ ──────→ │ Discriminador│
│             │         │  (compara)   │
└─────────────┘         └─────────────┘
```

- O **Gerador** aprende a distribuição conjunta dos dados
- O **Discriminador** tenta distinguir real vs. falso
- Após treino: Gerador produz dados estatisticamente similares aos reais

---

## 📁 Estrutura do Projeto

```
seminario/
├── tst.py                          # Script principal
├── README.md                       # Este arquivo
├── comparativo_ctgan_faker.png     # Gráfico gerado
├── relatorio_stack_e_*.json        # Relatório detalhado
└── requirements.txt                # Dependências
```

---

## 📚 Referências Bibliográficas

1. **XU, L. et al.** Modeling Tabular Data using Conditional GAN. *NeurIPS*, 2019.  
   → Artigo original do CTGAN. Disponível: https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html

2. **ANPD.** Guia de Privacidade e Proteção de Dados Pessoais. 2024.  
   → Contexto LGPD para dados sintéticos.

3. **SDV - Synthetic Data Vault.** MIT, 2024.  
   → Documentação: https://sdv.dev/

4. **GOODFELLOW, I. et al.** Generative Adversarial Networks. 2014.  
   → Artigo fundador das GANs.

---

## 💡 Dicas para Apresentação

### Slide 1: Problema
> "Como gerar dados sintéticos realistas sem violar privacidade?"

### Slide 2: Abordagens
- **Faker:** Regras manuais, rápido, mas não preserva padrões
- **CTGAN:** IA que aprende padrões, mais lento, mas fiel aos dados reais

### Slide 3: Demonstração
Rode o script e mostre os gráficos comparativos.

### Slide 4: Conclusão
> "CTGAN supera Faker em fidelidade estatística, sendo ideal para pesquisa e conformidade LGPD."

---

## ⚠️ Limitações

- CTGAN requer tempo de treinamento (não é instantâneo)
- Em datasets muito pequenos (<100 linhas), pode haver overfitting
- Não é adequado para dados com relações temporais complexas (use RNNs)

---
