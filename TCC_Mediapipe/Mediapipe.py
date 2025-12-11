import cv2
import numpy as np
import mediapipe as mp
import os

# --- CONFIGURAÇÃO ---
home = os.path.expanduser("~")
IMAGE_PATH = os.path.join("Y:\\Imagens\\minha_foto209.jpg")  # <-- Defina o caminho para sua imagem
SCALE_FACTOR = 0.5  # <-- Aumentado para melhor qualidade (teste entre 0.5 e 1.0)
# --------------------

# Utilitários do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Função que calcula o ângulo entre 3 pontos (a-b-c)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

class PostureAnalyzer:
    def __init__(self, smoothing_window=3):
        self.smoothing_window = smoothing_window
        self.angle_history = []
    
    def smooth_angle(self, angle):
        self.angle_history.append(angle)
        if len(self.angle_history) > self.smoothing_window:
            self.angle_history.pop(0)
        return np.mean(self.angle_history)
    
    def calculate_shoulder_pelvis_diff(self, landmarks, img_width, img_height):
        # Landmarks para ombros e quadris
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Converter para coordenadas de pixel
        ls_y = left_shoulder.y * img_height
        rs_y = right_shoulder.y * img_height
        lh_y = left_hip.y * img_height
        rh_y = right_hip.y * img_height
        
        # Calcular diferenças
        shoulder_diff = abs(ls_y - rs_y)
        hip_diff = abs(lh_y - rh_y)
        
        return {
            'shoulder_height_diff': shoulder_diff,
            'hip_height_diff': hip_diff,
            'shoulder_imbalance': (
                'Direito mais alto' if ls_y > rs_y else
                'Esquerdo mais alto' if ls_y < rs_y else
                'Altura igual'
            ),
            'hip_imbalance': (
                'Direito mais alto' if lh_y > rh_y else
                'Esquerdo mais alto' if lh_y < rh_y else
                'Altura igual'
            ),
            'left_shoulder_y': ls_y,
            'right_shoulder_y': rs_y,
            'left_hip_y': lh_y,
            'right_hip_y': rh_y
        }

def validate_landmark_quality(landmarks, min_visibility=0.7):
    key_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER, 
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    
    visibilities = [landmarks[point.value].visibility for point in key_points]
    return min(visibilities) >= min_visibility

def draw_reference_line(image, point1, point2, color, thickness=2):
    h, w = image.shape[:2]
    p1 = (int(point1[0] * w), int(point1[1] * h))
    p2 = (int(point2[0] * w), int(point2[1] * h))
    cv2.line(image, p1, p2, color, thickness)

# --- LEITURA E PROCESSAMENTO DA IMAGEM ---
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Erro: Não foi possível carregar a imagem em {IMAGE_PATH}")
    exit()

# Redimensiona a imagem mantendo melhor qualidade
if SCALE_FACTOR < 1.0:
    image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)

# Inicializar analisador
analyzer = PostureAnalyzer()

# Configuração otimizada do MediaPipe
with mp_pose.Pose(
    static_image_mode=True,       
    model_complexity=2,           # Aumentado para melhor precisão
    enable_segmentation=False,
    min_detection_confidence=0.7, # Aumentado para mais consistência
    min_tracking_confidence=0.7   
) as pose:

    frame = image.copy()

    # Converte para RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Processa a detecção de pose
    results = pose.process(image_rgb)

    # Volta para BGR para exibição
    image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_display.flags.writeable = True

    h, w = image_display.shape[:2]

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Valida qualidade da detecção
        landmark_quality = validate_landmark_quality(lm)
        
        # ANÁLISE COMPLETA DE POSTURA
        posture_data = analyzer.calculate_shoulder_pelvis_diff(lm, w, h)

        
        #Início alterações para altura
        altura_real = float(input("Digite a altura total da pessoa (em cm): "))


        # Considerando que a imagem vai até a pelve (~55% da altura total)
        altura_visivel = altura_real * 0.55  

        # Mede a altura visível detectada pelo MediaPipe (nariz até pelve)
        head_y = lm[mp_pose.PoseLandmark.NOSE.value].y * h
        pelvis_y = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].y + lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 * h
        altura_detectada_px = abs(pelvis_y - head_y)

        # Fator de conversão: cm por pixel
        fator_px_cm = altura_visivel / altura_detectada_px if altura_detectada_px > 0 else 0

        # Converter as diferenças dos ombros e quadris para cm
        shoulder_diff_cm = posture_data['shoulder_height_diff'] * fator_px_cm
        hip_diff_cm = posture_data['hip_height_diff'] * fator_px_cm

        # Exibir no console
        print("\n=== DIFERENÇAS CONVERTIDAS ===")
        print(f"Diferença entre ombros: {shoulder_diff_cm:.2f} cm")
        print(f"Diferença entre quadris: {hip_diff_cm:.2f} cm")

        # Categorizar simetria
        def categorizar_diferenca(diff_cm):
            if diff_cm < 1:
                return "Simétrico"
            elif diff_cm < 2:
                return "Assimetria leve"
            else:
                return "Assimetria significativa"

        cat_ombros = categorizar_diferenca(shoulder_diff_cm)
        cat_quadris = categorizar_diferenca(hip_diff_cm)

        print(f"Classificação ombros: {cat_ombros}")
        print(f"Classificação quadris: {cat_quadris}")
        
        #Fim alterações para altura

        # ÂNGULO DO QUADRIL
        shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        
        angle = calculate_angle(shoulder, hip, knee)
        smoothed_angle = analyzer.smooth_angle(angle)
        
        # Converter posições para pixels
        hip_px = tuple((np.array(hip) * [w, h]).astype(int))
        
        # EXIBIR TODAS AS INFORMAÇÕES
        info_lines = [
            f"ANGULO QUADRIL: {int(round(smoothed_angle))}°",
            f"DIFERENCA OMBROS: {shoulder_diff_cm:.2f} cm ({cat_ombros})",
            f"DIFERENCA QUADRIS: {hip_diff_cm:.2f} cm ({cat_quadris})",
            f"INCLINACAO OMBROS: {posture_data['shoulder_imbalance']}",
            f"INCLINACAO QUADRIS: {posture_data['hip_imbalance']}",
            f"QUALIDADE DETECCAO: {'ALTA' if landmark_quality else 'BAIXA'}"
        ]
        
        # Cores para o texto (verde para boa qualidade, amarelo para baixa)
        text_color = (0, 255, 0) if landmark_quality else (0, 255, 255)
        
        # Desenhar texto na imagem
        y_offset = 40
        for i, line in enumerate(info_lines):
            color = text_color if i < len(info_lines) - 1 else ((0, 255, 0) if landmark_quality else (0, 255, 255))
            cv2.putText(image_display, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        # Desenhar linhas de referência para visualização
        left_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                         lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                    lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                     lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        # Linhas horizontais de referência
        draw_reference_line(image_display, left_shoulder, right_shoulder, (0, 255, 0), 2)  # Verde para ombros
        draw_reference_line(image_display, left_hip, right_hip, (255, 0, 0), 2)  # Azul para quadris
        
        # Desenhar pontos-chave destacados
        key_points = [
            (left_shoulder, (0, 255, 0), "O-E"),
            (right_shoulder, (0, 255, 0), "O-D"), 
            (left_hip, (255, 0, 0), "Q-E"),
            (right_hip, (255, 0, 0), "Q-D")
        ]
        
        for point, color, label in key_points:
            px_point = (int(point[0] * w), int(point[1] * h))
            cv2.circle(image_display, px_point, 8, color, -1)
            cv2.putText(image_display, label, (px_point[0] + 10, px_point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Desenha esqueleto completo
        mp_drawing.draw_landmarks(
            image_display,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # Exibir ângulo próximo ao quadril
        cv2.putText(image_display, f"{int(round(smoothed_angle))}°",
                    hip_px, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Se qualidade baixa, adicionar aviso
        if not landmark_quality:
            cv2.putText(image_display, "AVISO: Use imagem com melhor iluminacao e fundo contrastante", 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        
    else:
        cv2.putText(image_display, "POSE NAO DETECTADA - Tente outra imagem", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostra o resultado
    scale_percent = 50  # Reduz para 50% do tamanho original
    width = int(image_display.shape[1] * scale_percent / 100)
    height = int(image_display.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(image_display, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Analise Postural - MediaPipe", resized)
    
    # Salvar resultado (opcional)
    output_path = "analise_postural_resultado.jpg"
    cv2.imwrite(output_path, image_display)
    print(f"Resultado salvo como: {output_path}")
    
    # Exibir resumo no console
    if results.pose_landmarks:
        print("\n=== RESUMO DA ANALISE POSTURAL ===")
        print(f"Angulo do quadril: {int(round(smoothed_angle))}°")
        print(f"Diferença entre ombros: {posture_data['shoulder_height_diff']:.1f} pixels")
        print(f"Diferença entre quadris: {posture_data['hip_height_diff']:.1f} pixels") 
        print(f"Inclinação dos ombros: {posture_data['shoulder_imbalance']}")
        print(f"Inclinação dos quadris: {posture_data['hip_imbalance']}")
        print(f"Qualidade da detecção: {'ALTA' if landmark_quality else 'BAIXA - Considere melhorar a imagem'}")

    # Espera até que uma tecla seja pressionada
    cv2.waitKey(0)

cv2.destroyAllWindows()