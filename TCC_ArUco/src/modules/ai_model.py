from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from typing import List, Union, Tuple, Dict
import sys
import os

DATASET_FILE = "../dataset/dataset_assimetria.csv" 

class SistemaDeTriagem:
    """
    Contém três modelos de classificação (Decision Tree) para a triagem 
    de assimetria em cada segmento corporal.
    """
    def __init__(self):
        """Inicializa os três modelos Decision Tree."""
        self.modelo_ombro = DecisionTreeClassifier(random_state=42)
        self.modelo_escapula = DecisionTreeClassifier(random_state=42)
        self.modelo_pelve = DecisionTreeClassifier(random_state=42)

    def treinar(self, X: pd.DataFrame, Y_ombro: List[str], Y_escapula: List[str], Y_pelve: List[str]):
        """
        Treina cada modelo com as features e a label específica do segmento.
        """
        if X.empty:
            raise ValueError("O dataset de treinamento não pode estar vazio.")
        
        X_ombro = X[['ombro_dif_Y']]
        self.modelo_ombro.fit(X_ombro, Y_ombro)
        print(f"Modelo Ombro treinado com {len(X_ombro)} amostras.")
        
        X_escapula = X[['escapula_dif_Y']]
        self.modelo_escapula.fit(X_escapula, Y_escapula)
        print(f"Modelo Escápula treinado com {len(X_escapula)} amostras.")
        
        X_pelve = X[['pelve_dif_Y']]
        self.modelo_pelve.fit(X_pelve, Y_pelve)
        print(f"Modelo Pelve treinado com {len(X_pelve)} amostras.")


    def prever(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Classifica as 3 assimetrias separadamente.
        
        Args:
            features: Dicionário contendo as 3 diferenças em cm.
            
        Returns:
            Dicionário com as 3 classificações.
        """
        
        predicao = {}
        
        X_ombro_pred = pd.DataFrame([features['Ombros']], columns=['ombro_dif_Y'])
        predicao['Ombros'] = self.modelo_ombro.predict(X_ombro_pred)[0]
        
        X_escapula_pred = pd.DataFrame([features['Escapulas']], columns=['escapula_dif_Y'])
        predicao['Escapulas'] = self.modelo_escapula.predict(X_escapula_pred)[0]
        
        X_pelve_pred = pd.DataFrame([features['Pelve']], columns=['pelve_dif_Y'])
        predicao['Pelve'] = self.modelo_pelve.predict(X_pelve_pred)[0]
        
        return predicao


def carregar_dataset_csv(caminho_arquivo: str) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Carrega o dataset de treinamento de um arquivo CSV com as 3 labels separadas.
    """
    print(f"\nTentando carregar dataset de: {os.path.abspath(caminho_arquivo)}")
    
    try:
        df = pd.read_csv(caminho_arquivo)
    except FileNotFoundError:
        sys.stderr.write(f"Erro: Arquivo CSV não encontrado em '{caminho_arquivo}'. Verifique o caminho.\n")
        return pd.DataFrame(), [], [], []
    except pd.errors.EmptyDataError:
        sys.stderr.write("Erro: O arquivo CSV está vazio.\n")
        return pd.DataFrame(), [], [], []
    except Exception as e:
        sys.stderr.write(f"Erro ao ler CSV: {e}\n")
        return pd.DataFrame(), [], [], []
        
    colunas_esperadas = ['ombro_dif_Y', 'escapula_dif_Y', 'pelve_dif_Y', 
                         'ombro_label', 'escapula_label', 'pelve_label']
    if not all(coluna in df.columns for coluna in colunas_esperadas):
        sys.stderr.write(f"Erro: O CSV deve conter as colunas: {colunas_esperadas}\n")
        return pd.DataFrame(), [], [], []
        
    try:
        X = df[['ombro_dif_Y', 'escapula_dif_Y', 'pelve_dif_Y']].astype(float)
        
        Y_ombro = df['ombro_label'].astype(str).tolist()
        Y_escapula = df['escapula_label'].astype(str).tolist()
        Y_pelve = df['pelve_label'].astype(str).tolist()
        
        print(f"Dataset carregado com sucesso! Total de {len(X)} amostras.")
        return X, Y_ombro, Y_escapula, Y_pelve
        
    except ValueError as e:
        sys.stderr.write(f"Erro: Falha na conversão de tipos de dados. Verifique se as colunas numéricas contêm apenas números. Detalhe: {e}\n")
        return pd.DataFrame(), [], [], []