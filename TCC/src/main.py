import cv2
import sys
import numpy as np
from typing import Optional, Dict
from modules.ai_model import *
from modules.aruco_processor import *
from config import *

caminho_foto_teste = '../Images/FotoLarissa.jpeg'

def _obter_input_flutuante(prompt: str) -> float:
    """Solicita um número flutuante do usuário com um valor padrão."""
    while True:
        entrada = input(f"{prompt}")
        try:
            valor = float(entrada.replace(',', '.'))
            return valor
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

def exibir_imagem(frame: np.ndarray, nome_janela: str = "Análise Corporal - TCC"):
    """Exibe a imagem redimensionada com tratamento de teclado."""
    
    largura_max = 1366
    altura_max = 768
    h, w = frame.shape[:2]
    escala = min(largura_max / w, altura_max / h)
    
    if escala < 1.0:
        novo_w = int(w * escala)
        novo_h = int(h * escala)
        frame_exibicao = cv2.resize(frame, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
    else:
        frame_exibicao = frame 

    cv2.namedWindow(nome_janela, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(nome_janela, frame_exibicao)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def executar_analise(caminho_foto: str):
    
    frame = cv2.imread(caminho_foto)
    if frame is None:
        sys.stderr.write(f"Erro: Não foi possível carregar a imagem em: {caminho_foto}\n")
        return

    medida_ombros_cm = _obter_input_flutuante(
        "Digite a medida real em cm entre os marcadores: "
    )
    
    X, Y_ombro, Y_escapula, Y_pelve = carregar_dataset_csv(DATASET_FILE)

    if X.empty:
        sys.stderr.write("Não foi possível treinar o modelo. Saindo da análise.\n")
        return
    
    classificador = SistemaDeTriagem()
    classificador.treinar(X, Y_ombro, Y_escapula, Y_pelve) 
    processor = ProcessadorArUco()
    frame_processado: Optional[np.ndarray] = None
    
    try:
        frame_processado = processor.detectar_e_processar(frame.copy())
        
        if frame_processado is None:
            print("Erro: Nenhum marcador ArUco detectado na imagem.")
            exibir_imagem(frame, "Análise: Sem Marcadores")
            return
            
        processor.calibrar_escala(medida_ombros_cm)
        print(f"Fator de conversão estabelecido: {processor.fator_cm_por_px:.4f} cm/pixel")
        
        resultados_assimetria = processor.analisar_assimetrias()
        classificacoes_segmentos: Dict[str, str] = {}
        
        if None in resultados_assimetria.values():
            classificacoes_segmentos = {
                'Ombros': 'Nao Classificado', 
                'Escapulas': 'Nao Classificado', 
                'Pelve': 'Nao Classificado'
            }
            print("Aviso: Pelo menos um par de marcadores está faltando. Classificação IA ignorada.")
        else:
            classificacoes_segmentos = classificador.prever({
                'Ombros': resultados_assimetria['Ombros'], 
                'Escapulas': resultados_assimetria['Escapulas'], 
                'Pelve': resultados_assimetria['Pelve']
            })

        frame_final = processor.desenhar_linhas_e_overlay(
            frame_processado, 
            resultados_assimetria, 
            classificacoes_segmentos, 
            medida_ombros_cm
        )
        
        exibir_imagem(frame_final)

    except (ValueError, ZeroDivisionError) as e:
        sys.stderr.write(f"\nErro Crítico no Processamento/Calibração: {e}\n")
        
        frame_erro = frame_processado or frame 
        cv2.putText(frame_erro, "Erro Crítico (Ver Console)", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)
        exibir_imagem(frame_erro, "Análise: Erro Crítico")
    
    except Exception as e:
        sys.stderr.write(f"\nErro Inesperado: {e}\n")
        exibir_imagem(frame, "Análise: Erro Inesperado")


if __name__ == '__main__':
    executar_analise(caminho_foto_teste)