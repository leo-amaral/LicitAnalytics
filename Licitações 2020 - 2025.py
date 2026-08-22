# LicitAnalytics

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys
sys.path.append(r'C:\Users\leo_a\AppData\Local\Programs\Python\Python313\Lib\site-packages')

# Importação dos arquivos das Licitações de 2020 a 2025
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


# Exportação do arquivo final em CSV e XLSL
df_unificado.to_csv( "Licitações 2020 - 2025.csv", index=False, sep=';', encoding='latin1'
)

df_unificado.to_excel("Licitações 2020 - 2025 Excel.xlsx", index=False)


'''
Modelo de Regressão Linear Simples para Previsão de Valores das Licitações
Nessa fase, o modelo preditivo utiliza apenas os Valores Previstos anuais anteriores das Licitações
'''

# Alterar os dois aqui de uma vez para ficar bonitinho nos gráficos e prints
termos_pesquisa = ['Merenda', r'perec[ií]ve']
termos_print = "Merenda Perecível"

'''
Alguns outros objetos que podem ser pesquisados acima
Em alguns casos, pode ser necessário usar Regex para que a pesquisa ocorra de forma correta
REMUME -> para medicamentos contidos na REMUME de Riolândia/SP
Cestas Básicas
Informática 
'''
mascara_objeto = pd.Series(True, index=df_unificado.index)

# Aplica o filtro rodando cada termo da lista separadamente na base
for termo in termos_pesquisa:
    mascara_objeto &= df_unificado['Objeto'].str.contains(termo, case=False, regex=True, na=False)

termo_exclusao = r"n[ãa]o" # Para excluir alguns termos do objeto caso necessário
mascara_objeto &= ~df_unificado['Objeto'].str.contains(termo_exclusao, case=False, regex=True, na=False)

# Filtra o objeto e remove linhas com Valor Total zero (para não distorcer a modelagem)
df_analise_valores = df_unificado[mascara_objeto & (df_unificado['Valor Total Licitação'] > 0)].copy()

# Trava: Verifica se encontrou dados antes de tentar modelar
if df_analise_valores.empty:
    print(f"\n[AVISO] Nenhum dado encontrado contendo todas as palavras: {termos_print}. Pulando Regressão de Valores.")
else:
    # Estratégia para tratamento de anomalias
    tratar_anomalia_valores = False # True para tratar e False para não tratar

    if tratar_anomalia_valores:
        df_treino_valores = df_analise_valores[df_analise_valores['Exercício'] != 22].copy() # Exercício é tratado apenas pelos últimos dígitos (20, 21, 22, 23, 24, 25)
        print("\nAnomalias tratadas - Exercício 2022 excluído")
    else:
        df_treino_valores = df_analise_valores.copy()
        print("\nNão foram constatadas anomalias - Série Histórica completa")
    # Correção: Transformação do exercício para Ano (Exercício Real) para os gráficos 
    df_anual_valores = df_treino_valores.groupby('Exercício')['Valor Previsto'].sum().reset_index()
    df_anual_valores['Exercício_Real'] = df_anual_valores['Exercício'] + 2000

    if not df_anual_valores.empty:
        X = df_anual_valores[['Exercício_Real']].values
        y = df_anual_valores['Valor Previsto'].values

        modelo = LinearRegression()
        modelo.fit(X, y)
        
        #Previsão para 2026
        previsao_valores_2026 = modelo.predict(np.array([[2026]]))
        previsao_2026 = max(0, previsao_valores_2026[0])
        r_quadrado_regressao_simples_valores = modelo.score(X, y)

        print(f"\n--- ANÁLISE PREDITIVA: {termos_print} ---")
        print(f"Previsão de valores para 2026: R$ {previsao_2026:,.2f}")

        # Gráfico
        plt.figure(figsize=(10,6))
        plt.scatter(df_anual_valores['Exercício_Real'], y, color='blue', label='Dados Históricos', s=100)

        x_linha = np.array([2020, 2026]).reshape(-1, 1)
        y_linha = modelo.predict(x_linha)
        plt.plot(x_linha, y_linha, color='red', linestyle='--', label='Tendência Linear')
        plt.scatter(2026, previsao_2026, color='green', marker='X', s=200, label=f'Projeção 2026: \nR$ {previsao_2026:,.2f}')

        plt.text(2020.2, y.max() * 1.4 , f'$R^2 = {r_quadrado_regressao_simples_valores:.4f}$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.ticklabel_format(style='plain', axis='y')
        plt.ylim(0, y.max() * 1.5)
        plt.xlim(2019, 2027)
        # plt.title(f'Regressão Linear: Valores Previstos de {texto_filtro_print}')
        plt.title(f'Regressão Linear Simples: Valores Previstos de {termos_print}')
        plt.xlabel('Ano')
        plt.ylabel('Valor Previsto (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

'''
Modelo de Regressão Linear Simples para Previsão de Economicidade das Licitações
Nessa fase, o modelo preditivo utiliza apenas a Economicidade das Licitações
O filtro utilizado aqui é o mesmo dos valores previstos
'''

df_analise_economicidade = df_unificado[mascara_objeto & (df_unificado['Valor Total Licitação'] > 0)].copy()

# Trava: Verifica se encontrou dados antes de tentar modelar
if df_analise_economicidade.empty:
    print(f"\n[AVISO] Nenhum dado encontrado para analisar economicidade. Pulando esta etapa.")
else:
    # Estratégia para tratamento de anomalias
    tratar_anomalia_economicidade = False # True para tratar e False para não tratar

    if tratar_anomalia_economicidade:
        df_treino_economicidade = df_analise_economicidade[df_analise_economicidade['Exercício'] != 23].copy() # Exercício é tratado apenas pelos últimos dígitos (20, 21, 22, 23, 24, 25)
        print("\nAnomalias tratadas - Exercício 2023 excluído")
    else:
        df_treino_economicidade = df_analise_economicidade.copy()
        print("\nNão foram constatadas anomalias - Série Histórica completa")
     
    df_anual_economicidade = df_treino_economicidade.groupby('Exercício').agg({
    'Valor Previsto': 'sum',
    'Economia Absoluta': 'sum'
    }).reset_index()

    # Calculando a taxa global do ano
    df_anual_economicidade['Economia Percentual'] = np.where(
        df_anual_economicidade['Valor Previsto'] > 0,
        df_anual_economicidade['Economia Absoluta'] / df_anual_economicidade['Valor Previsto'],
        0
    )
    # Correção: Transformação do exercício para Ano (Exercício Real) para os gráficos
    df_anual_economicidade['Exercício_Real'] = df_anual_economicidade['Exercício'] + 2000

    if not df_anual_economicidade.empty:
        X = df_anual_economicidade[['Exercício_Real']].values
        y = df_anual_economicidade['Economia Percentual'].values

        modelo = LinearRegression()
        modelo.fit(X, y)
        #Previsão para 2026
        previsao_economicidade_2026 = modelo.predict(np.array([[2026]]))
        previsao_2026_economicidade = max(0, previsao_economicidade_2026[0])
        r_quadrado_regressao_simples_economicidade = modelo.score(X, y)

        print(f"\n--- ANÁLISE PREDITIVA DE ECONOMICIDADE: {termos_print} ---")
        print(f"Previsão de economicidade para 2026: {(previsao_2026_economicidade * 100):,.2f}%")

        # Gráfico
        plt.figure(figsize=(10,6))
        plt.scatter(df_anual_economicidade['Exercício_Real'], y, color='blue', label='Dados Históricos', s=100)

        x_linha = np.array([2020, 2026]).reshape(-1, 1)
        y_linha = modelo.predict(x_linha)
        plt.plot(x_linha, y_linha, color='red', linestyle='--', label='Tendência Linear')
        plt.scatter(2026, previsao_2026_economicidade, color='green', marker='X', s=200, label=f'Projeção 2026: \n{(previsao_2026_economicidade * 100):,.2f}%')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.text(2020.2, y.max() * 1.4 , f'$R^2 = {r_quadrado_regressao_simples_economicidade:.4f}$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.ylim(0, y.max() * 1.5)
        plt.xlim(2019, 2027)
        # plt.title(f'Regressão Linear: Economicidade de {texto_filtro_print}')
        plt.title(f'Regressão Linear Simples: Economicidade de {termos_print}')
        plt.xlabel('Ano')
        plt.ylabel('Economicidade Percentual (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

'''
Modelo de Regressão Linear Múltipla para Previsão de Valores das Licitações
Nessa fase, o modelo preditivo utiliza também o indíce IPCA para estimar os Valores Previstos anuais das Licitações
O filtro utilizado aqui é o mesmo dos valores previstos para a comparação entre os modelos preditivos
'''

# Taxas Anuais do IPCA
ipca_historico = {
    2020: 4.52,
    2021: 10.06,
    2022: 5.79,
    2023: 4.62,
    2024: 4.83,
    2025: 4.26
}

# O modelo precisa do IPCA estimado de 2026 para fazer a previsão
ipca_2026_projecao = 5.02 # Extraído do Relatório de Mercado - Focus de 14/08/2026

if not df_anual_valores.empty:
    # Mapeando a inflação para os anos correspondentes
    df_anual_valores['IPCA'] = df_anual_valores['Exercício_Real'].map(ipca_historico)
    
    # X é uma matriz 2D (Tempo + Inflação)
    X_valores = df_anual_valores[['Exercício_Real', 'IPCA']].values
    y_valores = df_anual_valores['Valor Previsto'].values

    modelo_valores = LinearRegression()
    modelo_valores.fit(X_valores, y_valores)
    
    # Previsão para 2026 
    previsao_raw_2026 = modelo_valores.predict(np.array([[2026, ipca_2026_projecao]]))
    previsao_2026 = max(0, previsao_raw_2026[0])
    r_quadrado_regressao_multipla_valores = modelo_valores.score(X_valores, y_valores)

    print(f"\n--- ANÁLISE PREDITIVA MÚLTIPLA (ANO + IPCA): {termos_print} ---")
    print(f"Previsão de Valores para 2026: R$ {previsao_2026:,.2f}")

    # Gráfico adaptado para 2D para visualização
    plt.figure(figsize=(10,6))
    plt.scatter(df_anual_valores['Exercício_Real'], y_valores, color='blue', label='Dados Históricos', s=100)

    # Para plotar a linha ajustada em 2D, usamos as predições do próprio modelo histórico
    y_pred_historico = modelo_valores.predict(X_valores)
    # Linha completa para ser plotado até a previsão de 2026
    x_linha_completa = list(df_anual_valores['Exercício_Real']) + [2026]
    y_linha_completa = list(y_pred_historico) + [previsao_2026]
    plt.plot(x_linha_completa, y_linha_completa, 
             color='purple', 
             linestyle='--', 
             marker='s',     
             markersize=6,
             linewidth=2,
             label='Ajuste Múltiplo (Ano + IPCA)')
    
    plt.scatter(2026, previsao_2026, color='green', marker='X', s=200, label=f'Projeção 2026:\nR$ {previsao_2026:,.2f}')

    plt.text(2020.2, y_valores.max() * 1.4 , f'$R^2 = {r_quadrado_regressao_multipla_valores:.4f}$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylim(0, y_valores.max() * 1.5)
    plt.xlim(2019, 2027)
    plt.title(f'Regressão Linear Múltipla (Fator IPCA): Valores Previstos de {termos_print}')
    plt.xlabel('Ano')
    plt.ylabel('Valor Previsto (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

'''
Modelo de Regressão Linear Múltipla para Previsão de Economicidade das Licitações
Nessa fase, o modelo preditivo utiliza também o indíce IPCA para estimar a Economicidade de 2026 das Licitações
O filtro utilizado aqui é o mesmo da Economicidade para a comparação entre os modelos preditivos
'''

if not df_anual_economicidade.empty:
    # Mapeando a inflação para os anos correspondentes
    df_anual_economicidade['IPCA'] = df_anual_economicidade['Exercício_Real'].map(ipca_historico)
    
    # X é uma matriz 2D (Tempo + Inflação)
    X_economicidade = df_anual_economicidade[['Exercício_Real', 'IPCA']].values
    y_economicidade = df_anual_economicidade['Economia Percentual'].values

    modelo_economicidade = LinearRegression()
    modelo_economicidade.fit(X_economicidade, y_economicidade)
    
    # Previsão para 2026 
    previsao_raw_2026_economicidade = modelo_economicidade.predict(np.array([[2026, ipca_2026_projecao]]))
    previsao_2026_economicidade = max(0, previsao_raw_2026_economicidade[0])
    r_quadrado_regressao_multipla_economicidade = modelo_economicidade.score(X_economicidade, y_economicidade)

    print(f"\n--- ANÁLISE PREDITIVA MÚLTIPLA (ANO + IPCA): {termos_print} ---")
    print(f"Previsão de Economicidade para 2026: {(previsao_2026_economicidade * 100):,.2f}%")

    # Gráfico adaptado para 2D para visualização
    plt.figure(figsize=(10,6))
    plt.scatter(df_anual_economicidade['Exercício_Real'], y_economicidade, color='blue', label='Dados Históricos', s=100)

    # Para plotar a linha ajustada em 2D, usamos as predições do próprio modelo histórico
    y_pred_historico_economicidade = modelo_economicidade.predict(X_valores)
    # Linha completa para ser plotado até a previsão de 2026
    x_linha_completa = list(df_anual_economicidade['Exercício_Real']) + [2026]
    y_linha_completa = list(y_pred_historico_economicidade) + [previsao_2026_economicidade]
    plt.plot(x_linha_completa, y_linha_completa, 
             color='purple', 
             linestyle='--', 
             marker='s',     
             markersize=6,
             linewidth=2,
             label='Ajuste Múltiplo (Ano + IPCA)')
    
    plt.scatter(2026, previsao_2026_economicidade, color='green', marker='X', s=200, label=f'Projeção 2026:\n{(previsao_2026_economicidade * 100):,.2f}%')

    plt.text(2020.2, y_economicidade.max() * 1.4 , f'$R^2 = {r_quadrado_regressao_multipla_economicidade:.4f}$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylim(0, y_economicidade.max() * 1.5)
    plt.xlim(2019, 2027)
    plt.title(f'Regressão Linear Múltipla (Fator IPCA): Economicidade de {termos_print}')
    plt.xlabel('Ano')
    plt.ylabel('Economicidade Prevista (%)')
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
    'Local', 'Funcional', 'Cód. de aplicação', 'Descrição do Cód. de aplicação', 
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
df_empenhos_unificado.to_csv("Despesas Gerais 2020 - 2025.csv", sep=';', index=False, encoding='latin1', decimal=',')