# Corte de carga com fluxo de potência AC no IEEE 14

Código em Python para minimizar o corte de carga após a retirada de ramos do sistema IEEE de 14 barras. Os dados elétricos são do [caso base da PGLib-OPF v23.07](https://github.com/power-grid-lib/pglib-opf/blob/master/pglib_opf_case14_ieee.m), sem aumento artificial das demandas. A base de potência é 100 MVA.

## Como executar

Com Python instalado, abra o PowerShell nesta pasta e execute:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Os dados do sistema e os parâmetros do cenário estão no início de `main.py`. Não é necessário baixar outro arquivo de dados.

## Cenário apresentado no resumo

O cenário configurado em `main.py` retira somente o ramo **1–2** (`RAMOS_FORA = [(1, 2)]`). A função objetivo minimiza o corte ativo ponderado pelos fatores de prioridade. Os fatores são 1,05 na barra 3; 1,015 na barra 4; 1,02 nas barras 9, 10, 11 e 14; 1,01 nas barras 2, 5, 6, 12 e 13; e 1,00 nas demais. Esses valores foram definidos para o estudo e não representam uma classificação real de consumidores.

Antes da contingência, o sistema atende os **259 MW** de demanda sem corte. Após a retirada de 1–2, a melhor solução factível encontrada entre dois pontos iniciais apresenta **82,2859 MW** de corte (31,7706% da demanda) e **função objetivo ponderada de 0,845876 pu**. O corte físico e a função objetivo são medidas diferentes, pois os fatores de prioridade entram apenas na segunda. Os resultados detalhados estão em `resultados/ramos_1-2/resultado.json`.

## Modelo e resultados

O programa utiliza balanços de potência ativa e reativa do fluxo AC, redespacho da geração, corte proporcional das demandas ativa e reativa, taps e shunts fixos e limites de geração, tensão, diferença angular e potência aparente nas duas extremidades de cada ramo. A solução é obtida com `scipy.optimize.minimize(method="SLSQP")`. O programa verifica os resíduos e limites com tolerância de `1e-5` pu (ou rad para ângulos). Como o problema AC não é convexo, o resultado é uma solução local factível, sem garantia de ótimo global.

Ao executar `main.py`, o programa salva um `resultado.json` e quatro figuras na pasta do cenário:

```text
resultados/ramos_1-2/
├── resultado.json
└── figuras/
    ├── 01_carga_antes_depois.png
    ├── 02_geracao_antes_depois.png
    ├── 03_ramos_antes_depois.png
    └── 04_tensoes_antes_depois.png
```

## Testar outros ramos

Para mudar a contingência, edite `RAMOS_FORA` no início de `main.py`. Cada par deve seguir a ordem `(de, para)` da tabela `RAMOS`. Por exemplo, `[(3, 4)]` retira um ramo, `[(1, 2), (2, 3)]` retira dois e `[]` mantém a rede intacta. Cada configuração grava seus resultados em uma pasta própria dentro de `resultados/`. Os fatores de prioridade podem ser alterados no vetor `ALFA`. Para interpretar um novo cenário, confira o `resultado.json` e as verificações de viabilidade; a solução local pode mudar com o ponto inicial.
