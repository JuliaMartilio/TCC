import cv2
import numpy as np
import sys
from typing import Dict, Optional, List, Tuple
from config import MARCADOR_IDS, PARES_DE_ANALISE, ARUCO_DICT

class ProcessadorArUco:
    """
    Gerencia a detecção de marcadores ArUco, calibração de escala 
    e análise de assimetria.
    """
    def __init__(self):
        """Inicializa o detector ArUco."""
        self.arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict)
        self.marcadores: Dict[int, np.ndarray] = {}
        self.fator_cm_por_px: float = 0.0

    def _calcular_centro(self, corners: np.ndarray) -> np.ndarray:
        """Calcula o centro (cx, cy) de um marcador a partir de seus cantos."""
        c = corners[0].astype(np.float32)
        cx = np.mean(c[:, 0])
        cy = np.mean(c[:, 1])
        return np.array([cx, cy])

    def detectar_e_processar(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detecta marcadores na imagem, calcula seus centros e desenha o overlay.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        self.marcadores = {}

        LABEL_SCALE = 1.5          
        LABEL_COLOR = (255, 255, 255) 
        LABEL_THICKNESS = 2        
        LABEL_OUTLINE_THICKNESS = LABEL_THICKNESS + 2 
        LABEL_OUTLINE_COLOR = (0, 0, 0) 

        if ids is not None:
            chaves_validas = set(MARCADOR_IDS.keys()) 
            
            for i, id_raw in enumerate(ids.flatten()):
                id_int = int(id_raw) 
                
                if id_int in chaves_validas: 
                    centro = self._calcular_centro(corners[i])
                    self.marcadores[id_int] = centro 

                    cv2.circle(frame, tuple(map(int, centro)), 6, (0, 255, 0), -1)

                    cv2.putText(frame, MARCADOR_IDS[id_int],
                                (int(centro[0]), int(centro[1]) - 15),
                                cv2.FONT_HERSHEY_DUPLEX, LABEL_SCALE, LABEL_OUTLINE_COLOR, LABEL_OUTLINE_THICKNESS)

                    cv2.putText(frame, MARCADOR_IDS[id_int],
                                (int(centro[0]), int(centro[1]) - 15),
                                cv2.FONT_HERSHEY_DUPLEX, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS)
                        
            return frame
        
        return None

    def calibrar_escala(self, medida_ombros_cm: float):
        """Calcula o fator de conversão de pixel para cm usando a distância dos ombros."""
        
        if 1 in self.marcadores and 2 in self.marcadores:
            p1 = self.marcadores[1]  
            p2 = self.marcadores[2] 
            
            dist_px = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            if dist_px > 0:
                self.fator_cm_por_px = medida_ombros_cm / dist_px
            else:
                raise ZeroDivisionError("Distância em pixels entre os ombros é zero.")
        else:
            raise ValueError("Marcadores de Ombro (ID 1 e 2) são necessários para a calibração de escala.")
            
    def analisar_assimetrias(self) -> Dict[str, Optional[float]]:
        """
        Calcula as diferenças de altura (Y) em cm para os três pares.
        Retorna um dicionário {Nome_Par: Diferenca_cm}.
        """
        resultados = {}
        
        for id_esq, id_dir, nome_par, _ in PARES_DE_ANALISE:
            if id_esq in self.marcadores and id_dir in self.marcadores:
                p_esq = self.marcadores[id_esq]
                p_dir = self.marcadores[id_dir]
                
                dif_px = abs(p_esq[1] - p_dir[1])
                
                dif_cm = dif_px * self.fator_cm_por_px
                resultados[nome_par] = dif_cm
            else:
                resultados[nome_par] = None
                
        return resultados

    def desenhar_linhas_e_overlay(self, frame: np.ndarray, resultados_assimetria: Dict[str, Optional[float]], classificacoes_segmentos: Dict[str, str], medida_ombros_cm: float) -> np.ndarray:
        """Desenha as linhas de referência, a legenda e a classificação na imagem."""
        
        LINE_THICKNESS = 5 
        
        for id_esq, id_dir, _, cor in PARES_DE_ANALISE:
            if id_esq in self.marcadores and id_dir in self.marcadores:

                cv2.line(frame, tuple(map(int, self.marcadores[id_esq])),
                         tuple(map(int, self.marcadores[id_dir])),
                         cor, LINE_THICKNESS) 
                         
        y0 = 60 
        
        TEXT_SCALE = 2.0
        TEXT_COLOR = (255, 255, 255) 
        TEXT_THICKNESS = 4 
        TEXT_OUTLINE_THICKNESS = TEXT_THICKNESS + 2
        TEXT_OUTLINE_COLOR = (0, 0, 0) 
        LINE_SPACING = 60
        CLASSIFICACAO_TEXT_COLOR = (0, 255, 255) 

        overlay_textos_topo = [
            f"Fator: {self.fator_cm_por_px:.4f} cm/pixel",
            f"Ombros real: {medida_ombros_cm:.2f} cm",
            "--- Diferencas (Y) e CLASSIFICACAO em cm ---",
        ]
        
        def formatar_linha(nome_par, dif_cm):
            dif_str = f"{dif_cm:.2f}" if dif_cm is not None else "Ausente"
            classif_str = classificacoes_segmentos.get(nome_par, 'ERRO').upper()
            
            return f"{nome_par}: {dif_str} cm | {classif_str}"

        overlay_textos_classificacao = [
            formatar_linha('Ombros', resultados_assimetria.get('Ombros')),
            formatar_linha('Escapulas', resultados_assimetria.get('Escapulas')),
            formatar_linha('Pelve', resultados_assimetria.get('Pelve')),
        ]
        
        for texto in overlay_textos_topo:
            cv2.putText(frame, texto, (10, y0),
                        cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, TEXT_OUTLINE_COLOR, TEXT_OUTLINE_THICKNESS)

            cv2.putText(frame, texto, (10, y0),
                        cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS) 
            y0 += LINE_SPACING
            
        for texto in overlay_textos_classificacao:
            cv2.putText(frame, texto, (10, y0),
                        cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, TEXT_OUTLINE_COLOR, TEXT_OUTLINE_THICKNESS)

            cv2.putText(frame, texto, (10, y0),
                        cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, CLASSIFICACAO_TEXT_COLOR, TEXT_THICKNESS) 
            y0 += LINE_SPACING
            
        return frame