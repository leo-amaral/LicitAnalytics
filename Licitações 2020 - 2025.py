# Licitações 2020 - 2025
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\leo_a\AppData\Local\Programs\Python\Python313\Lib\site-packages')

# Importação dos arquivos
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
        .str.replace(".", "", regex=False)      # milhar
        .str.replace(",", ".", regex=False)     # decimal
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

# Modelo de Regressão Linear para Previsão de Economicidade das Licitações

# Parâmetro de análise - Poderá ser facilmente alterado para escalabilidade do projeto
filtro_objeto = 'Cestas Básicas'

# Filtra o objeto e remove linhas com Valor Total zero (para não distorcer a modelagem)
df_analise = df_unificado[
    (df_unificado['Objeto'].str.contains(filtro_objeto, case=False, na=False)) & 
    (df_unificado['Valor Total Licitação'] > 0)
].copy()

# Estratégia para tratamento de anomalias
tratar_anomalia = False # True para tratar alguma anomalia e False para não tratar

if tratar_anomalia:
    df_treino = df_analise[df_analise['Exercício'] != 23].copy() # Exercício é tratado apenas pelos últimos dígitos (20, 21, 22, 23, 24, 25)
    print("Anomalias tratadas - Exercício 2022 excluído")
else:
    df_treino = df_analise.copy()
    print("Não foram constatadas anomalias - Série Histórica completa")

df_anual = df_treino.groupby('Exercício')['Valor Total Licitação'].sum().reset_index()

# Correção: Transformação do exercício para Ano (Exercício Real)
df_anual['Exercício_Real'] = df_anual['Exercício'] + 2000

# Regressão Linear usando os anos corretos
X = df_anual[['Exercício_Real']].values
y = df_anual['Valor Total Licitação'].values

modelo = LinearRegression()
modelo.fit(X, y)

# Previsão para 2026
previsao_bruta_2026 = modelo.predict(np.array([[2026]]))
previsao_2026 = max(0, previsao_bruta_2026[0])

r_quadrado = modelo.score(X, y)

print(f"\n--- ANÁLISE PREDITIVA: {filtro_objeto} ---")
print(f"Previsão de gasto para 2026: R$ {previsao_2026:,.2f}")

# Gráfico
plt.figure(figsize=(10,6))
plt.scatter(df_anual['Exercício_Real'], y, color='blue', label='Dados Históricos', s=100)

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