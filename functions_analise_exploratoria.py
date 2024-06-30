## Bibliotecas de Análise de Dados
import pandas as pd 
import builtins as builtins
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display, Image
from tabulate import tabulate
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Bibliotecas de Manipulação de Tempo
from datetime import datetime, date

## Bibliotecas de Modelagem Matemática e Estatística
import numpy as np
import scipy as sp 
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import normaltest, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, uniform, chi2_contingency
from statsmodels.stats.weightstats import ztest
from numpy import interp
import random

# Bibliotecas de Seleção de Modelos
from skopt import BayesSearchCV
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif

# Bibliotecas de Pré-Processamento e Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Bibliotecas de Modelos de Machine Learning
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Bibliotecas de Métricas de Machine Learning
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score, f1_score, log_loss, brier_score_loss, confusion_matrix, silhouette_score

def plota_barras(lista_variaveis, hue, df, linhas, colunas, titulo, rotation):        

    k = 0
    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', color='#1FB3E5')
    ax.set_title(f'{titulo}')
    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
    ax.set_ylabel(f'Quantidade', fontsize = 14)
    total = []
    for bar in ax.patches:
        height = bar.get_height()
        total.append(height)
    total = builtins.sum(total)
    
    sizes = []
    for bar in ax.patches:
        height = bar.get_height()
        sizes.append(height)
        ax.text(bar.get_x() + bar.get_width()/1.6,
                height,
                f'{builtins.round((height/total)*100, 2)}%',
                ha = 'center',
                fontsize = 12
        )
    ax.set_ylim(0, builtins.max(sizes)*1.1)
    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
    # Formatação manual dos rótulos do eixo y para remover a notação científica
    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
    # Adicionamos os nomes das categorias no eixo x
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
    plt.show()

def plota_grafico_linhas(df, x, y, title):

    # Criando o gráfico de linha
    plt.figure(figsize=(14, 7))
    plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

    # Adicionando títulos e rótulos aos eixos
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

    for i, txt in enumerate(df[y]):
        plt.annotate(f'{txt:.2f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Exibindo o gráfico
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def analisa_distribuicao_via_percentis(df, variaveis):
    def highlight_percentiles(s):
        is_1_percentile = s.name == '1%'
        is_99_8_percentile = s.name == '99.8%'
        if is_1_percentile or is_99_8_percentile:
            return ['background-color: blue'] * len(s)
        else:
            return [''] * len(s)

    percentis = df[variaveis].describe(percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.998]).style.apply(highlight_percentiles, axis=1)    

    return percentis


def analisa_correlacao(metodo, df):
    plt.figure(figsize=(10, 5))
    mask = np.triu(np.ones_like(df.corr(method=metodo), dtype=bool))
    heatmap = sns.heatmap(df.corr(method=metodo), vmin=-1, vmax=1, cmap='magma', annot = True, cbar_kws={"shrink": .8}, mask = mask)
    heatmap.set_title(f"Analisando Correlação de {metodo}")
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

def compara_medias_amostras(df, variavel):

    def teste_hipotese_z_duas_amostras_independentes(amostra1, amostra2, variavel):
        print(f"Hipótese Nula (H0): A média amostral do público com Churn e do público sem Churn são iguais para a variável '{variavel}'.")
        print(f"Hipótese Alternativa (H1): A média amostral do público com Churn e do público sem Churn são diferentes para a variável '{variavel}'.")
        print()
        
        z_stat, p_value = ztest(amostra1, amostra2)

        alpha = 0.05
        alpha_bicaudal = alpha / 2  

        if p_value < alpha_bicaudal:
            print(f'Pelo Teste de Hipótese Z bicaudal, o p-valor foi de {p_value:.4f}, logo, rejeita-se H0.')
        else:
            print(f'Pelo Teste de Hipótese Z bicaudal, o p-valor foi de {p_value:.4f}, logo, não rejeita-se H0.')


    # Cria separa entre quem é Churn e quem não é Churn
    df_eda_continuas_com_churn = df_eda_continuas.loc[df_eda_continuas["churn"] == 1]
    df_eda_continuas_sem_churn = df_eda_continuas.loc[df_eda_continuas["churn"] == 0]

    # Aplica o Teorema do Limite Central, gerando duas distribuições de médias amostras que aproximam-se de uma Normal
    medias_amostrais_com_churn = []
    medias_amostrais_sem_churn = []

    for i in range(10000):
        amostra_churn = random.choices(df_eda_continuas_com_churn[variavel].values, k=1000)
        media_amostra_churn = np.mean(amostra_churn)
        medias_amostrais_com_churn.append(media_amostra_churn)

        amostra_sem_churn = random.choices(df_eda_continuas_sem_churn[variavel].values, k=1000)
        media_amostra_sem_churn = np.mean(amostra_sem_churn)
        medias_amostrais_sem_churn.append(media_amostra_sem_churn)

    teste_hipotese_z_duas_amostras_independentes(medias_amostrais_com_churn, medias_amostrais_sem_churn, variavel)

    # Plotando os histogramas sobrepostos
    plt.figure(figsize = (8,4))
    plt.hist(medias_amostrais_com_churn, bins=30, alpha=0.5, label='Churn', linewidth=5, color = "red")  # alpha controla a transparência
    plt.hist(medias_amostrais_sem_churn, bins=30, alpha=0.5, label='sem Churn', linewidth=5, color = "green")
    plt.legend()  # Mostra a legenda com os rótulos
    plt.xlabel('Valores')
    plt.ylabel('Frequência')
    plt.title(f'Distribuição das Médias Amostrais de "{variavel}" ')
    plt.grid(True)
    plt.show()


def woe(df, feature, target):
    churn = df.loc[df[target] == 1].groupby(feature, as_index = False)[target].count().rename({target:'churn'}, axis = 1)
    sem_churn = df.loc[df[target] == 0].groupby(feature, as_index = False)[target].count().rename({target:'sem_churn'}, axis = 1)

    woe = churn.merge(sem_churn, on = feature, how = 'left')
    woe['percent_churn'] = woe['churn']/woe['churn'].sum()
    woe['percent_sem_churn'] = woe['sem_churn']/woe['sem_churn'].sum()
    woe['woe'] = round(np.log(woe['percent_churn']/woe['percent_sem_churn']), 3)
    woe.sort_values(by = 'woe', ascending = True, inplace = True)
    weight_of_evidence = woe['woe'].unique()


    woe['woe'] = round(np.log(woe['percent_churn'] / woe['percent_sem_churn']), 3)
    woe['iv'] = ((woe['percent_churn'] - woe['percent_sem_churn']) * np.log(woe['percent_churn'] / woe['percent_sem_churn'])).sum()
    woe.sort_values(by='woe', ascending=True, inplace=True)

    x = list(woe[feature])
    y = list(woe['woe'])

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='o', linestyle='--', linewidth=2, color='#1FB3E5')

    for label, value in zip(x, y):
        plt.text(x=label, y=value, s=str(value), fontsize=10, color='red', ha='left', va='center', rotation=45)

    plt.title(f'Weight of Evidence da variável "{feature}"', fontsize=14)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Weight of Evidence', fontsize=14)
    plt.xticks(ha='right', fontsize=10, rotation=45)
    plt.show()
