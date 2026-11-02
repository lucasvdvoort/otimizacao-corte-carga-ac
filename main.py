"""Corte minimo de carga com fluxo AC no sistema IEEE de 14 barras.

Dados: PGLib-OPF v23.07, pglib_opf_case14_ieee.m (caso base).
https://github.com/power-grid-lib/pglib-opf/blob/master/pglib_opf_case14_ieee.m

As potencias e admitancias sao calculadas na base de 100 MVA. A demanda
e a do caso original; a contingencia retira os ramos em RAMOS_FORA.
O solver SLSQP encontra candidatos locais; a viabilidade e conferida.
"""

from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# ----------------------------- Dados e cenario -----------------------------
BASE_MVA = 100.0
RAMOS_FORA = [(1, 2)]  # contingencia estudada; pode ser alterada
TOL_VIABILIDADE = 1e-5  # pu (ou rad para a diferenca angular)


def pasta_resultados():
    """Uma pasta para cada conjunto de ramos retirados."""
    nome = "_".join(f"{de}-{para}" for de, para in RAMOS_FORA) or "nenhum"
    pasta = Path(__file__).resolve().parent / "resultados" / f"ramos_{nome}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta

# Barra, Pd [MW], Qd [MVAr], Gs [MW], Bs [MVAr], Vmin, Vmax.
BARRAS = np.array([
    [1, 0.0, 0.0, 0.0, 0.0, 0.94, 1.06],
    [2, 21.7, 12.7, 0.0, 0.0, 0.94, 1.06],
    [3, 94.2, 19.0, 0.0, 0.0, 0.94, 1.06],
    [4, 47.8, -3.9, 0.0, 0.0, 0.94, 1.06],
    [5, 7.6, 1.6, 0.0, 0.0, 0.94, 1.06],
    [6, 11.2, 7.5, 0.0, 0.0, 0.94, 1.06],
    [7, 0.0, 0.0, 0.0, 0.0, 0.94, 1.06],
    [8, 0.0, 0.0, 0.0, 0.0, 0.94, 1.06],
    [9, 29.5, 16.6, 0.0, 19.0, 0.94, 1.06],
    [10, 9.0, 5.8, 0.0, 0.0, 0.94, 1.06],
    [11, 3.5, 1.8, 0.0, 0.0, 0.94, 1.06],
    [12, 6.1, 1.6, 0.0, 0.0, 0.94, 1.06],
    [13, 13.5, 5.8, 0.0, 0.0, 0.94, 1.06],
    [14, 14.9, 5.0, 0.0, 0.0, 0.94, 1.06],
], dtype=float)

# Peso de prioridade alpha_i por barra (1 a 14)
# Um peso maior torna o corte nessa barra mais caro para o otimizador
ALFA = np.array([1, 1.01, 1.05, 1.015, 1.01, 1.01, 1, 1, 1.02, 1.02, 1.02, 1.01, 1.01, 1.02], dtype=float)  # barra 3 prioritária

# Barra, Pg inicial [MW], Qg inicial [MVAr], Qmax, Qmin, Pmax, Pmin.
GERADORES = np.array([
    [1, 170.0, 5.0, 10.0, 0.0, 340.0, 0.0],
    [2, 29.5, 0.0, 30.0, -30.0, 59.0, 0.0],
    [3, 0.0, 20.0, 40.0, 0.0, 0.0, 0.0],
    [6, 0.0, 9.0, 24.0, -6.0, 0.0, 0.0],
    [8, 0.0, 9.0, 24.0, -6.0, 0.0, 0.0],
], dtype=float)

# De, para, r [pu], x [pu], b total [pu], limite [MVA], tap,
# angmin e angmax [graus]
RAMOS = np.array([
    [1, 2, .01938, .05917, .0528, 472, 0, -30, 30],
    [1, 5, .05403, .22304, .0492, 128, 0, -30, 30],
    [2, 3, .04699, .19797, .0438, 145, 0, -30, 30],
    [2, 4, .05811, .17632, .0340, 158, 0, -30, 30],
    [2, 5, .05695, .17388, .0346, 161, 0, -30, 30],
    [3, 4, .06701, .17103, .0128, 160, 0, -30, 30],
    [4, 5, .01335, .04211, .0000, 664, 0, -30, 30],
    [4, 7, .00000, .20912, .0000, 141, .978, -30, 30],
    [4, 9, .00000, .55618, .0000, 53, .969, -30, 30],
    [5, 6, .00000, .25202, .0000, 117, .932, -30, 30],
    [6, 11, .09498, .19890, .0000, 134, 0, -30, 30],
    [6, 12, .12291, .25581, .0000, 104, 0, -30, 30],
    [6, 13, .06615, .13027, .0000, 201, 0, -30, 30],
    [7, 8, .00000, .17615, .0000, 167, 0, -30, 30],
    [7, 9, .00000, .11001, .0000, 267, 0, -30, 30],
    [9, 10, .03181, .08450, .0000, 325, 0, -30, 30],
    [9, 14, .12711, .27038, .0000, 99, 0, -30, 30],
    [10, 11, .08205, .19207, .0000, 141, 0, -30, 30],
    [12, 13, .22092, .19988, .0000, 99, 0, -30, 30],
    [13, 14, .17093, .34802, .0000, 76, 0, -30, 30],
], dtype=float)

N_BARRAS = len(BARRAS)
if len(ALFA) != N_BARRAS or np.any(ALFA <= 0):
    raise ValueError("ALFA precisa ter um peso positivo para cada barra.")
IND_GER = GERADORES[:, 0].astype(int) - 1
IND_PG = np.flatnonzero(GERADORES[:, 5] > GERADORES[:, 6])
IND_CARGA = np.flatnonzero(BARRAS[:, 1] > 0)
N_C = len(IND_CARGA)
N_PG = len(IND_PG)
N_QG = len(GERADORES)

# x = [cortes | geracao ativa | geracao reativa | tensoes | angulos 2..14]
C_INI = 0
PG_INI = C_INI + N_C
QG_INI = PG_INI + N_PG
V_INI = QG_INI + N_QG
TH_INI = V_INI + N_BARRAS
N_VAR = TH_INI + N_BARRAS - 1


def componentes(dados):
    """Grupos de barras que continuam conectadas pelos ramos em servico."""
    vizinhos = [[] for _ in range(N_BARRAS)]
    for de, para in dados[:, :2].astype(int):
        vizinhos[de - 1].append(para - 1)
        vizinhos[para - 1].append(de - 1)
    grupos, vistos = [], set()
    for inicio in range(N_BARRAS):
        if inicio in vistos:
            continue
        grupo, pilha = [], [inicio]
        vistos.add(inicio)
        while pilha:
            i = pilha.pop()
            grupo.append(i)
            for j in vizinhos[i]:
                if j not in vistos:
                    vistos.add(j)
                    pilha.append(j)
        grupos.append(grupo)
    return grupos


def preparar_ramos(retirar_contingencia):
    """Equacao (4): admitancia serie e tap dos ramos em servico."""
    mask = np.ones(len(RAMOS), dtype=bool)
    if retirar_contingencia:
        for de, para in RAMOS_FORA:
            mask &= ~((RAMOS[:, 0] == de) & (RAMOS[:, 1] == para))
    dados = RAMOS[mask]
    i = dados[:, 0].astype(int) - 1
    j = dados[:, 1].astype(int) - 1
    y = 1 / (dados[:, 2] + 1j * dados[:, 3])
    tap = np.where(dados[:, 6] == 0, 1.0, dados[:, 6])
    grupos = componentes(dados)
    fontes = set(GERADORES[GERADORES[:, 5] > 0, 0].astype(int) - 1)
    # Ilha sem gerador ativo fica desenergizada e perde toda a sua carga.
    desenergizadas = {i for grupo in grupos if not fontes.intersection(grupo)
                     for i in grupo}
    ativas = np.array([i for i in range(N_BARRAS) if i not in desenergizadas])
    return {
        "dados": dados, "i": i, "j": j,
        "grupos": grupos, "desenergizadas": desenergizadas, "ativas": ativas,
        "y": y, "b": dados[:, 4], "tap": tap,
        "rate": dados[:, 5] / BASE_MVA,
        "angmin": np.deg2rad(dados[:, 7]),
        "angmax": np.deg2rad(dados[:, 8]),
    }


def separar(x):
    """Converte o vetor real do solver nas variaveis fisicas do modelo."""
    c = np.zeros(N_BARRAS)
    c[IND_CARGA] = x[C_INI:PG_INI]
    pg = np.zeros(N_QG)
    pg[IND_PG] = x[PG_INI:QG_INI]
    qg = x[QG_INI:V_INI]
    v = x[V_INI:TH_INI]
    theta = np.r_[0.0, x[TH_INI:N_VAR]]  # barra 1: referencia angular
    return c, pg, qg, v, theta


def calcular_rede(x, ramos):
    """Equacoes (4)-(8): tensoes, correntes e potencias nas duas pontas."""
    _, _, _, v, theta = separar(x)
    tensao = v * np.exp(1j * theta)
    vi, vj = tensao[ramos["i"]], tensao[ramos["j"]]
    y, b, tap = ramos["y"], ramos["b"], ramos["tap"]
    # (5)-(6): corrente que sai de i e corrente que sai de j; tap real.
    iij = (y + 1j * b / 2) / tap**2 * vi - y / tap * vj
    iji = -y / tap * vi + (y + 1j * b / 2) * vj
    # (7)-(8): S = V * conjugado(I), em cada extremidade do ramo.
    sij = vi * np.conj(iij)
    sji = vj * np.conj(iji)
    saidas = np.zeros(N_BARRAS, dtype=complex)
    np.add.at(saidas, ramos["i"], sij)
    np.add.at(saidas, ramos["j"], sji)
    shunt = v**2 * (BARRAS[:, 3] - 1j * BARRAS[:, 4]) / BASE_MVA
    return sij, sji, saidas + shunt


def construir_problema(retirar_contingencia, exigir_corte_zero=False):
    pd = BARRAS[:, 1].copy() / BASE_MVA
    qd = BARRAS[:, 2].copy() / BASE_MVA
    ramos = preparar_ramos(retirar_contingencia)
    desligadas = ramos["desenergizadas"]

    def objetivo(x):
        # Equacao (1): minimizar soma(alpha_i * C_i), com i nas barras de carga.
        return float(np.dot(ALFA[IND_CARGA], x[C_INI:PG_INI]))

    def balancos(x):
        c, pg, qg, _, _ = separar(x)
        # (2)-(3): carga ativa e reativa atendida; o fator de potencia e mantido.
        p_atendido = pd - c
        q_atendido = qd.copy()
        q_atendido[IND_CARGA] *= 1 - c[IND_CARGA] / pd[IND_CARGA]
        # (9)-(10): geracao - carga - fluxos - shunt = 0 em cada barra.
        injecao = -(p_atendido + 1j * q_atendido)
        np.add.at(injecao, IND_GER, pg + 1j * qg)
        _, _, saidas = calcular_rede(x, ramos)
        erro = injecao - saidas
        return np.r_[erro.real, erro.imag]

    def desigualdades(x):
        sf, st, _ = calcular_rede(x, ramos)
        _, _, _, _, theta = separar(x)
        dif = theta[ramos["i"]] - theta[ramos["j"]]
        # (14)-(15): limites MVA nas duas pontas e diferenca angular.
        # O SLSQP espera desigualdades na forma g(x) >= 0.
        return np.r_[
            ramos["rate"]**2 - abs(sf)**2,
            ramos["rate"]**2 - abs(st)**2,
            dif - ramos["angmin"],
            ramos["angmax"] - dif,
        ]

    limites = []
    # Limite do corte: 0 <= C_i <= P_D,i. Na pre-contingencia, C_i = 0.
    for i in IND_CARGA:
        if i in desligadas:
            limites.append((pd[i], pd[i]))  # ilha sem fonte: carga toda cortada
        else:
            limites.append((0.0, 0.0 if exigir_corte_zero else pd[i]))
    # (11)-(13): limites de geracao ativa, reativa e tensao.
    for g in IND_PG:
        limites.append((0.0, 0.0) if IND_GER[g] in desligadas else
                       (GERADORES[g, 6] / BASE_MVA, GERADORES[g, 5] / BASE_MVA))
    for g in range(N_QG):
        limites.append((0.0, 0.0) if IND_GER[g] in desligadas else
                       (GERADORES[g, 4] / BASE_MVA, GERADORES[g, 3] / BASE_MVA))
    for i in range(N_BARRAS):
        limites.append((0.0, 0.0) if i in desligadas else
                       (BARRAS[i, 5], BARRAS[i, 6]))
    # (15): theta_1 = 0; cada ilha energizada ganha uma referencia angular.
    referencias = {min(grupo) for grupo in ramos["grupos"]}
    limites.extend([(0.0, 0.0) if i in referencias or i in desligadas
                    else (None, None) for i in range(1, N_BARRAS)])
    return pd, qd, ramos, objetivo, balancos, desigualdades, limites


def ponto_inicial(pd, tipo="flat"):
    x = np.zeros(N_VAR)
    # Pg1 absorve a demanda nao coberta por Pg2; as perdas serao ajustadas.
    x[PG_INI:QG_INI] = [min(3.4, max(0.0, sum(pd) - 0.5)), 0.5]
    x[QG_INI:V_INI] = GERADORES[:, 2] / BASE_MVA
    x[V_INI:TH_INI] = 1.0
    if tipo == "alternativo":
        x[PG_INI:QG_INI] = [min(3.4, sum(pd)), 0.0]
        x[V_INI:TH_INI] = 1.02
    return x


def conferir(x, problema):
    pd, _, ramos, _, balancos, _, limites = problema
    sf, st, _ = calcular_rede(x, ramos)
    _, pg, qg, v, theta = separar(x)
    erro = balancos(x)
    dif = theta[ramos["i"]] - theta[ramos["j"]]
    lim_inf = np.array([(-np.inf if a is None else a) for a, _ in limites])
    lim_sup = np.array([(np.inf if b is None else b) for _, b in limites])
    corte = float(np.sum(x[C_INI:PG_INI]) * BASE_MVA)
    corte_ponderado = float(np.dot(ALFA[IND_CARGA], x[C_INI:PG_INI]))
    geracao = float(np.sum(pg) * BASE_MVA)
    perdas = float(np.sum((sf + st).real) * BASE_MVA)
    atendida = float(np.sum(pd) * BASE_MVA - corte)
    ativas = ramos["ativas"]
    violacao_ramo = (max(0, np.max(abs(sf) - ramos["rate"]),
                         np.max(abs(st) - ramos["rate"])) if len(sf) else 0)
    violacao_angular = (max(0, np.max(ramos["angmin"] - dif),
                            np.max(dif - ramos["angmax"])) if len(dif) else 0)
    return {
        "max_balanco_p_pu": float(np.max(abs(erro[:N_BARRAS]))),
        "max_balanco_q_pu": float(np.max(abs(erro[N_BARRAS:]))),
        "max_limite_variavel_pu": float(max(0, np.max(lim_inf - x), np.max(x - lim_sup))),
        "max_limite_ramo_pu": float(violacao_ramo),
        "max_limite_angular_rad": float(violacao_angular),
        "corte_mw": corte,
        "objetivo_ponderado_pu": corte_ponderado,
        "geracao_mw": geracao,
        "perdas_mw": perdas,
        "demanda_atendida_mw": atendida,
        "erro_balanco_total_p_pu": abs(geracao - atendida - perdas) / BASE_MVA,
        "v_min_pu": float(np.min(v[ativas])),
        "v_max_pu": float(np.max(v[ativas])),
        "barras_desenergizadas": [i + 1 for i in sorted(ramos["desenergizadas"])],
        "qg_mvar": [float(q * BASE_MVA) for q in qg],
    }


def e_factivel(checks):
    return all(checks[k] <= TOL_VIABILIDADE for k in (
        "max_balanco_p_pu", "max_balanco_q_pu", "max_limite_variavel_pu",
        "max_limite_ramo_pu", "max_limite_angular_rad", "erro_balanco_total_p_pu",
    ))


def resolver(problema, x0):
    _, _, ramos, objetivo, balancos, desigualdades, limites = problema
    barras_ativas = ramos["ativas"]
    x0 = x0.copy()
    # Mudar o zero angular de uma ilha nao altera seus fluxos internos.
    theta0 = np.r_[0.0, x0[TH_INI:N_VAR]]
    for grupo in ramos["grupos"]:
        referencia = theta0[min(grupo)]
        for i in grupo:
            if i > 0:
                x0[TH_INI + i - 1] -= referencia
    for k, (minimo, maximo) in enumerate(limites):
        if minimo is not None and minimo == maximo:
            x0[k] = minimo

    def balancos_solver(x):
        erro = balancos(x)
        # Barras desenergizadas tem balanco 0 = 0; conferir avalia todas depois.
        return np.r_[erro[barras_ativas], erro[N_BARRAS + barras_ativas]]

    restricoes = [{"type": "eq", "fun": balancos_solver}]
    if len(ramos["dados"]):
        restricoes.append({"type": "ineq", "fun": desigualdades})
    inicio = perf_counter()
    resultado = minimize(
        objetivo, x0, method="SLSQP", bounds=limites,
        constraints=restricoes,
        options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
    )
    tempo = perf_counter() - inicio
    checks = conferir(resultado.x, problema)
    return resultado, checks, tempo


def criar_graficos(x_pre, problema_pre, x_pos, problema_pos, pasta):
    pasta.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True})
    barras = BARRAS[:, 0].astype(int)
    c, pg, qg, v, _ = separar(x_pos)
    _, pg_pre, qg_pre, v_pre, _ = separar(x_pre)
    pd = problema_pos[0] * BASE_MVA

    # Figura 1: a coluna inteira e a demanda anterior; hachurado e o corte.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    corte_mw = c * BASE_MVA
    barras_cortadas = corte_mw > .01  # evita desenhar tracos de erro numerico
    ax.bar(barras, pd - c * BASE_MVA, width=.58,
           color="#287d8e", label="Carga atendida")
    ax.bar(barras[barras_cortadas], corte_mw[barras_cortadas],
           bottom=(pd - corte_mw)[barras_cortadas],
           width=.58, color="#C45152", edgecolor="white",
           hatch="///", label="Carga cortada")
    for i in IND_CARGA:
        if barras_cortadas[i]:
            ax.annotate(f"{corte_mw[i]:.2f} MW", (barras[i], pd[i]),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=8, color="#a3373e")
    ax.set(xlabel="Barra", ylabel="Potência ativa [MW]",
           title="Carga atendida e cortada após a contingência")
    ax.set_xticks(barras)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(pasta / "01_carga_antes_depois.png", dpi=250)
    plt.close(fig)

    # Figura 2: redespacho de P e Q dos cinco dispositivos de geracao.
    ger_barras = GERADORES[:, 0].astype(int)
    fig, eixos = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    for ax, anterior, posterior, unidade, titulo in (
        (eixos[0], pg_pre, pg, "MW", "Geracao ativa"),
        (eixos[1], qg_pre, qg, "MVAr", "Geracao reativa"),
    ):
        ax.bar(np.arange(N_QG) - .19, anterior * BASE_MVA, width=.38,
               color="#b8c0c5", label="Antes")
        ax.bar(np.arange(N_QG) + .19, posterior * BASE_MVA, width=.38,
               color="#287d8e", label="Depois")
        ax.set(ylabel=unidade, title=titulo)
        ax.grid(axis="y", alpha=.25)
    eixos[0].legend(frameon=False, ncol=2)
    eixos[1].set_xticks(np.arange(N_QG), [str(i) for i in ger_barras])
    eixos[1].set_xlabel("Barra do gerador")
    fig.suptitle("Redespacho após a contingencia")
    fig.tight_layout()
    fig.savefig(pasta / "02_geracao_antes_depois.png", dpi=250)
    plt.close(fig)

    # Figura 3: ramo retirado aparece apenas na coluna anterior.
    sf_pre, st_pre, _ = calcular_rede(x_pre, problema_pre[2])
    sf_pos, st_pos, _ = calcular_rede(x_pos, problema_pos[2])
    ramos_pre, ramos_pos = problema_pre[2], problema_pos[2]
    uso_pre = 100 * np.maximum(abs(sf_pre), abs(st_pre)) / ramos_pre["rate"]
    uso_pos = 100 * np.maximum(abs(sf_pos), abs(st_pos)) / ramos_pos["rate"]
    pos_por_ramo = {(int(a), int(b)): uso for (a, b), uso in
                    zip(ramos_pos["dados"][:, :2], uso_pos)}
    rotulos = [f"{int(a)}-{int(b)}" for a, b in ramos_pre["dados"][:, :2]]
    uso_depois = [pos_por_ramo.get((int(a), int(b)), 0.0)
                   for a, b in ramos_pre["dados"][:, :2]]
    y = np.arange(len(rotulos))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y - .19, uso_pre, height=.38, color="#b8c0c5", label="Antes")
    ax.barh(y + .19, uso_depois, height=.38, color="#287d8e", label="Depois")
    ax.axvline(100, color="#bc4b51", linestyle="--", linewidth=1)
    ax.set_yticks(y, rotulos)
    ax.invert_yaxis()
    ax.set(xlabel="Maior carregamento entre as extremidades [%]",
           ylabel="Ramo", title="Carregamento antes e depois da contingencia")
    for de, para in RAMOS_FORA:
        ax.annotate("retirado", (1, rotulos.index(f"{de}-{para}") + .19),
                    fontsize=8, va="center", color="#a3373e")
    ax.grid(axis="x", alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pasta / "03_ramos_antes_depois.png", dpi=250)
    plt.close(fig)

    # Figura 4: perfil de tensao e faixa permitida.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(barras, v_pre, marker="o", color="#8d979d", label="Antes")
    v_energizada = v.copy()
    desligadas = sorted(problema_pos[2]["desenergizadas"])
    v_energizada[desligadas] = np.nan
    ax.plot(barras, v_energizada, marker="s", color="#287d8e", label="Depois")
    if desligadas:
        ax.scatter(barras[desligadas], np.zeros(len(desligadas)), marker="x",
                   color="#bc4b51", s=70, label="Desenergizada", zorder=3)
    ax.axhline(.94, color="#bc4b51", linestyle="--", linewidth=1)
    ax.axhline(1.06, color="#bc4b51", linestyle="--", linewidth=1)
    piso = 0 if desligadas else .92
    ax.set(xlabel="Barra", ylabel="Magnitude da tensao [pu]",
           title="Tensoes antes e depois da contingencia", ylim=(piso, 1.08))
    ax.set_xticks(barras)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pasta / "04_tensoes_antes_depois.png", dpi=250)
    plt.close(fig)


def main():
    if len(RAMOS_FORA) != len(set(map(tuple, RAMOS_FORA))):
        raise ValueError("Ha ramos repetidos em RAMOS_FORA.")
    for de, para in RAMOS_FORA:
        existe = np.any((RAMOS[:, 0] == de) & (RAMOS[:, 1] == para))
        if not existe:
            raise ValueError(f"Ramo {de}-{para} nao existe em RAMOS.")
    historico_pre = []
    escolhido = None
    problema_pre = construir_problema(False, exigir_corte_zero=True)
    for indice, tipo in enumerate(("flat", "alternativo"), 1):
        res, check, tempo = resolver(problema_pre, ponto_inicial(problema_pre[0], tipo))
        historico_pre.append({"inicio": indice,
                              "success": bool(res.success), "message": str(res.message),
                              "tempo_s": tempo, "checks": check})
        if res.success and e_factivel(check) and check["corte_mw"] <= 1e-6:
            escolhido = res.x, check, tempo
            break
    if escolhido is None:
        raise RuntimeError("O caso original nao foi validado sem corte antes da contingencia.")

    x_pre, check_pre, tempo_pre = escolhido
    problema_pos = construir_problema(True)
    tentativas_pos = [resolver(problema_pos, x0) for x0 in
                     (x_pre, ponto_inicial(problema_pos[0], "alternativo"))]
    factiveis = [(k, res, check, tempo) for k, (res, check, tempo)
                in enumerate(tentativas_pos, 1) if res.success and e_factivel(check)]
    if not factiveis:
        detalhes = [(r.message, c) for r, c, _ in tentativas_pos]
        raise RuntimeError(f"Nenhuma solucao pos-contingencia validada: {detalhes}")
    # (1): entre candidatos factiveis, escolher o menor objetivo ponderado.
    indice, res, check, tempo_pos = min(factiveis, key=lambda item: item[1].fun)
    c, pg, qg, v, theta = separar(res.x)
    sf, st, _ = calcular_rede(res.x, problema_pos[2])
    ramos = problema_pos[2]
    demanda_mw = np.sum(problema_pos[0]) * BASE_MVA

    saida = {
        "fonte": "PGLib-OPF v23.07, pglib_opf_case14_ieee.m, caso base",
        "base_mva": BASE_MVA,
        "ramos_fora": [list(ramo) for ramo in RAMOS_FORA],
        "demanda_total_mw": demanda_mw,
        "fatores_prioridade_por_barra": ALFA.tolist(),
        "pre_contingencia": {
            "tempo_s": tempo_pre, "checks": check_pre,
            "corte_por_barra_mw": (separar(x_pre)[0] * BASE_MVA).tolist(),
            "pg_por_gerador_mw": (separar(x_pre)[1] * BASE_MVA).tolist(),
            "qg_por_gerador_mvar": (separar(x_pre)[2] * BASE_MVA).tolist(),
            "tensao_por_barra_pu": separar(x_pre)[3].tolist(),
        },
        "pos_contingencia": {"inicio_escolhido": indice, "tempo_s": tempo_pos,
                             "checks": check, "corte_percentual": 100 * check["corte_mw"] / demanda_mw,
                             "objetivo_ponderado_pu": float(res.fun),
                             "corte_por_barra_mw": (c * BASE_MVA).tolist(),
                             "pg_por_gerador_mw": (pg * BASE_MVA).tolist(),
                             "qg_por_gerador_mvar": (qg * BASE_MVA).tolist(),
                             "tensao_por_barra_pu": v.tolist(),
                             "angulo_por_barra_graus": np.rad2deg(theta).tolist(),
                             "ramos": [
                                 {"de": int(a), "para": int(b),
                                  "sf_mva": float(abs(s1) * BASE_MVA),
                                  "st_mva": float(abs(s2) * BASE_MVA),
                                  "carregamento_percentual": float(100 * max(abs(s1), abs(s2)) / lim)}
                                 for (a, b), s1, s2, lim in zip(ramos["dados"][:, :2], sf, st, ramos["rate"])
                             ]},
        "tentativas_pre": historico_pre,
        "tentativas_pos": [{"inicio": k, "success": bool(r.success),
                            "message": str(r.message), "tempo_s": t, "checks": ch}
                           for k, (r, ch, t) in enumerate(tentativas_pos, 1)],
        "nota": "PNL nao convexa: melhor solucao local factivel observada, sem garantia global.",
    }
    pasta = pasta_resultados()
    arquivo = pasta / "resultado.json"
    arquivo.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    criar_graficos(x_pre, problema_pre, res.x, problema_pos, pasta / "figuras")

    print(f"Demanda total: {demanda_mw:.3f} MW")
    prioridades = ", ".join(f"barra {i + 1}: {peso:g}" for i, peso in enumerate(ALFA)
                            if peso != 1) or "todas as barras com peso 1"
    print(f"Prioridades: {prioridades}")
    print(f"Ramos retirados: {RAMOS_FORA}")
    print(f"Pre-contingencia: corte zero, erros P/Q = {check_pre['max_balanco_p_pu']:.2e}/"
          f"{check_pre['max_balanco_q_pu']:.2e} pu")
    print(f"Pos-contingencia: corte = {check['corte_mw']:.6f} MW "
          f"({100 * check['corte_mw'] / demanda_mw:.3f}%), inicio {indice}")
    print(f"Indice de corte ponderado: {res.fun:.6f} pu")
    if check["barras_desenergizadas"]:
        print(f"Barras desenergizadas: {check['barras_desenergizadas']}")
    for i in IND_CARGA:
        if c[i] > 1e-7:
            print(f"Corte na barra {i + 1}: {c[i] * BASE_MVA:.6f} MW")
    print(f"Perdas ativas: {check['perdas_mw']:.6f} MW; "
          f"geracao: {check['geracao_mw']:.6f} MW")
    print(f"Tensoes: {check['v_min_pu']:.5f} a {check['v_max_pu']:.5f} pu")
    print(f"Maior violacao de balanco P/Q: {check['max_balanco_p_pu']:.2e}/"
          f"{check['max_balanco_q_pu']:.2e} pu")
    print("Convexidade: objetivo linear; conjunto factivel AC nao convexo.")
    print(f"Resultados: {arquivo}")


if __name__ == "__main__":
    main()