# Licitações 2020 - 2025

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys
sys.path.append(r'C:\Users\leo_a\AppData\Local\Programs\Python\Python313\Lib\site-packages')

# Importação dos arquivos das Licitações
arquivos_csv = [
    "Portal Transp. Licitações - 2020.csv",
    "Portal Transp. Licitações - 2021.csv",
    "Portal Transp. Licitações - 2022.csv",
    "Portal Transp. Licitações - 2023.csv",
    "Portal Transp. Licitações - 2024.csv",
    "Portal Transp. Licitações - 2025.csv"
]

dfs = [pd.read_csv(f, sep=';', encoding='latin1') for f in arquivos_csv]
df_unificado = pd.concat(dfs, ignore_index=True)

# Remoção das linhas que tenham a modalidade "LEILÃO"
if "Modalidade" in df_unificado.columns:
    df_unificado = df_unificado[df_unificado["Modalidade"].astype(str).str.strip().str.upper() != "LEILÃO"]

# Remoção de colunas que não serão utilizadas
colunas_drop = [
    "Data Inicio Proposta",
    "Data Fim Proposta",
    "Prazo de Entrega/Início",
    "Hora Abert. Propost."
]
df_unificado.drop(columns=colunas_drop, inplace=True, errors='ignore')

# Conversão dos valores monetários
def converter_moeda(coluna):
    return (
        coluna.astype(str)
        .str.replace("R$", "", regex=False)
        .str.replace("\xa0", "", regex=False)   # espaço invisível
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)      # remove ponto de milhar
        .str.replace(",", ".", regex=False)     # troca vírgula decimal por ponto
        .replace({"": None, "nan": None, "-": None})
        .pipe(pd.to_numeric, errors="coerce")
    )

for col in ["Valor Previsto", "Valor Total Licitação"]:
    df_unificado[col] = converter_moeda(df_unificado[col])

# Criando Economia Absoluta
df_unificado["Economia Absoluta"] = df_unificado["Valor Previsto"] - df_unificado["Valor Total Licitação"]

# Criando Economia Percentual (com tratamento para evitar divisão por zero)
df_unificado["Economia Percentual"] = np.where(
    df_unificado["Valor Previsto"] > 0,
    (df_unificado["Economia Absoluta"] / df_unificado["Valor Previsto"]),
    0
)

'''
# Exportação do arquivo final em CSV e XLSL
df_unificado.to_csv( "Portal_Transp_Licitacoes_Unificado.csv", index=False, sep=';', encoding='latin1'
)

df_unificado.to_excel("Portal_Transp_Licitacoes_Unificado.xlsx", index=False)
'''

# Modelo de Regressão Linear para Previsão de Valores das Licitações

# Parâmetro de análise - Poderá ser facilmente alterado para escalabilidade do projeto
filtro_objeto = 'Cestas Básicas'

# Filtra o objeto e remove linhas com Valor Total zero (para não distorcer a modelagem)
df_analise_valores = df_unificado[
    (df_unificado['Objeto'].str.contains(filtro_objeto, case=False, na=False)) & 
    (df_unificado['Valor Total Licitação'] > 0)
].copy()

# Estratégia para tratamento de anomalias
tratar_anomalia_valores = False # True para tratar alguma anomalia e False para não tratar

if tratar_anomalia_valores:
    df_treino_valores = df_analise_valores[df_analise_valores['Exercício'] != 23].copy() # Exercício é tratado apenas pelos últimos dígitos (20, 21, 22, 23, 24, 25)
    print("\nAnomalias tratadas - Exercício 2022 excluído")
else:
    df_treino_valores = df_analise_valores.copy()
    print("\nNão foram constatadas anomalias - Série Histórica completa")

df_anual_valores = df_treino_valores.groupby('Exercício')['Valor Total Licitação'].sum().reset_index()

# Correção: Transformação do exercício para Ano (Exercício Real)
df_anual_valores['Exercício_Real'] = df_anual_valores['Exercício'] + 2000

# Regressão Linear usando os anos corretos
X = df_anual_valores[['Exercício_Real']].values
y = df_anual_valores['Valor Total Licitação'].values

modelo = LinearRegression()
modelo.fit(X, y)

# Previsão para 2026
previsao_valores_2026 = modelo.predict(np.array([[2026]]))
previsao_2026 = max(0, previsao_valores_2026[0])

r_quadrado = modelo.score(X, y)

print(f"\n--- ANÁLISE PREDITIVA: {filtro_objeto} ---")
print(f"Previsão de gasto para 2026: R$ {previsao_2026:,.2f}")

# Gráfico
plt.figure(figsize=(10,6))
plt.scatter(df_anual_valores['Exercício_Real'], y, color='blue', label='Dados Históricos', s=100)

x_linha = np.array([2020, 2026]).reshape(-1, 1) # Linha de tendência
y_linha = modelo.predict(x_linha)
plt.plot(x_linha, y_linha, color='red', linestyle='--', label='Tendência Linear')
plt.scatter(2026, previsao_2026, color='green', marker='X', s=200, label='Projeção 2026')

plt.text(2020.2, y.max() * 1.4 , f'$R^2 = {r_quadrado:.4f}$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.ticklabel_format(style='plain', axis='y')
plt.ylim(0, y.max() * 1.5)
plt.xlim(2019, 2027)
plt.title('Regressão Linear: Gastos com Cestas Básicas')
plt.xlabel('Ano')
plt.ylabel('Valor Total Licitação (R$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Modelo de Regressão Linear para Previsão de Economicidade das Licitações
# A Princípio será utilizado o mesmo filtro_objetos para as duas previsões

# Filtra o objeto e remove linhas com Valor Total zero (para não distorcer a modelagem)
df_analise_economicidade = df_unificado[
    (df_unificado['Objeto'].str.contains(filtro_objeto, case=False, na=False)) & 
    (df_unificado['Valor Total Licitação'] > 0)
].copy()

# Estratégia para tratamento de anomalias
tratar_anomalia_economicidade = False # True para tratar alguma anomalia e False para não tratar

if tratar_anomalia_economicidade:
    df_treino_economicidade = df_analise_economicidade[df_analise_economicidade['Exercício'] != 23].copy() # Exercício é tratado apenas pelos últimos dígitos (20, 21, 22, 23, 24, 25)
    print("\nAnomalias tratadas - Exercício 2022 excluído")
else:
    df_treino_economicidade = df_analise_economicidade.copy()
    print("\nNão foram constatadas anomalias - Série Histórica completa")

df_anual_economicidade = df_treino_economicidade.groupby('Exercício')['Economia Percentual'].mean().reset_index()

# Correção: Transformação do exercício para Ano (Exercício Real)
df_anual_economicidade['Exercício_Real'] = df_anual_economicidade['Exercício'] + 2000

# Regressão Linear usando os anos corretos
X = df_anual_economicidade[['Exercício_Real']].values
y = df_anual_economicidade['Economia Percentual'].values

modelo = LinearRegression()
modelo.fit(X, y)

# Previsão para 2026
previsao_economicidade_2026 = modelo.predict(np.array([[2026]]))
previsao_2026_economicidade = max(0, previsao_economicidade_2026[0])

r_quadrado = modelo.score(X, y)

print(f"\n--- ANÁLISE PREDITIVA: {filtro_objeto} ---")
print(f"Previsão de economicidade para 2026: {(previsao_2026_economicidade * 100):,.2f}%")

# Gráfico
plt.figure(figsize=(10,6))
plt.scatter(df_anual_economicidade['Exercício_Real'], y, color='blue', label='Dados Históricos', s=100)

x_linha = np.array([2020, 2026]).reshape(-1, 1) # Linha de tendência
y_linha = modelo.predict(x_linha)
plt.plot(x_linha, y_linha, color='red', linestyle='--', label='Tendência Linear')
plt.scatter(2026, previsao_2026_economicidade, color='green', marker='X', s=200, label='Projeção 2026')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.text(2020.2, y.max() * 1.4 , f'$R^2 = {r_quadrado:.4f}$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.ylim(0, y.max() * 1.5)
plt.xlim(2019, 2027)
plt.title('Regressão Linear: Gastos com Cestas Básicas')
plt.xlabel('Ano')
plt.ylabel('Economicidade Percentual (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

'''
    Fase 2 do LicitAnalytics
    Importação dos Dados de Empenhos dos anos de 2020 a 2025
    Serão excluídos todos os empenhos não relacionados a licitações - Boa parte dos empenhos
    Depois do tratamento e unificação dos dados será exportado para o Power BI
    Não há previsão da implantação de modelos de regressão linear (por enquanto) para modelos preditivos
'''

arquivos_empenhos = [
    "Portal Transparencia Despesas Gerais - Exercício 2020.csv",
    "Portal Transparencia Despesas Gerais - Exercício 2021.csv",
    "Portal Transparencia Despesas Gerais - Exercício 2022.csv",
    "Portal Transparencia Despesas Gerais - Exercício 2023.csv",
    "Portal Transparencia Despesas Gerais - Exercício 2024.csv",
    "Portal Transparencia Despesas Gerais - Exercício 2025.csv"
]

dfs_empenhos = []

for arquivo in arquivos_empenhos:
    try:
        ano_exercicio = arquivo.split("Exercício ")[1].replace(".csv", "")
        df = pd.read_csv(arquivo, sep=';', encoding='latin1', low_memory=False, dtype={'Proc. Licitatório': str})
        df['Ano_Empenho'] = ano_exercicio
        dfs_empenhos.append(df)
    except Exception as e:
        print(f"Erro ao carregar {arquivo}: {e}")

# Unificação dos dados
df_empenhos_unificado = pd.concat(dfs_empenhos, ignore_index=True)

# Exclusão de todas essas colunas ai mesmo
colunas_para_excluir = [
    'Cód. Forn.', 'Nome Fornecedor', 'Nome Natureza', 'N° Ficha', 
    'Dotação', 'Alteração Dotação', 'Dotação Atual', 'Reforço', 
    'Empenhado até Hoje', 'Liquidado até Hoje', 'Pago até Hoje', 
    'Local', 'Funcional', 'Função', 'Nome da Função', 'Subfunção', 
    'Nome da Subfunção', 'Cód. de aplicação', 'Descrição do Cód. de aplicação', 
    'Fonte', 'Fonte de Recurso', 'Cód. Fonte', 'Código Fonte', 
    'Fonte STN', 'Nome Fonte STN'
]

df_empenhos_unificado.drop(columns=colunas_para_excluir, errors='ignore', inplace=True)

# Conversão dos valores monetários
colunas_monetarias = ['Valor Empenhado', 'Valor Anulado', 'Valor Liquidado', 'Valor Pago']

for col in colunas_monetarias:
    df_empenhos_unificado[col] = (
        df_empenhos_unificado[col].astype(str)
        .str.replace("R$", "", regex=False)
        .str.replace("\xa0", "", regex=False) # espaço invisível
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)    # remove ponto de milhar
        .str.replace(",", ".", regex=False)   # troca vírgula decimal por ponto
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    
df_empenhos_unificado['Empenho Líquido'] = df_empenhos_unificado['Valor Empenhado'] - df_empenhos_unificado['Valor Anulado']

df_empenhos_unificado = df_empenhos_unificado.dropna(subset=['Proc. Licitatório']) # Exclui todos os empenhos que não tem processo licitatório vinculado

# Exportação dos dados
# df_empenhos_unificado.to_csv("Despesas_Gerais_Unificado.csv", sep=';', index=False, encoding='latin1', decimal=',')