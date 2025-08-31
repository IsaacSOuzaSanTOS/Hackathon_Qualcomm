# Este script captura imagens da webcam, detecta bocejos em tempo real usando MediaPipe e um modelo PyTorch,
# e emite um alerta sonoro caso bocejos sejam detectados consecutivamente.

import time
from collections import deque
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import torch

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

t0 = time.perf_counter()

mouth_landmarks = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 42, 183, 78, 50, 280, 152, 377, 400, 164, 393
]


def play_wav(path: str = "alert.wav") -> None:
    """
    Reproduz um arquivo .wav de alerta sonoro.

    Args:
        path (str): Caminho para o arquivo .wav. Padrão é "alert.wav".
    """
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()


def centralizar_normalizar(pontos: np.ndarray) -> np.ndarray:
    """
    Centraliza e normaliza um conjunto de pontos 2D.

    Args:
        pontos (np.ndarray): Array de pontos (x, y).

    Returns:
        np.ndarray: Pontos centralizados e normalizados.
    """

    pontos = np.array(pontos)
    if pontos.ndim == 1:
        pontos = pontos.reshape(-1, 2)
    
    centroide = pontos.mean(axis=0)
    pontos_centralizados = pontos - centroide
    
    norma = np.linalg.norm(pontos_centralizados, axis=1).max()
    if norma == 0:
        norma = 1
    pontos_normalizados = pontos_centralizados / norma
    
    return np.array(pontos_normalizados)


def embedar_mapeamento(frame: np.ndarray) -> List[int]:
    """
    Extrai e retorna as coordenadas dos pontos da boca do rosto detectado no frame.

    Args:
        frame (np.ndarray): Imagem do frame da webcam.

    Returns:
        List[int]: Lista de coordenadas x, y dos pontos da boca.
    """

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if frame_rgb is None:
        print(f"Erro ao carregar a imagem")
        

    results = face_mesh.process(frame_rgb)
    coords = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame_rgb.shape
            for idx in mouth_landmarks:
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * width)
                y = int(lm.y * height)
                coords += [x, y]
                cv2.circle(frame_rgb, (x, y), 1, (0, 255, 0), -1)
                
    return coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.jit.load("models/yawn_model_full.ts", map_location=device)
model.eval()

@torch.no_grad()
def predict_proba(sample_np: np.ndarray) -> float:
    """
    Retorna a probabilidade prevista de bocejo para o vetor de entrada.

    Args:
        sample_np (np.ndarray): Vetor de características extraídas do rosto.

    Returns:
        float: Probabilidade de bocejo.
    """

    x = torch.tensor(sample_np, dtype=torch.float32).view(1, -1).to(device)
    logits = model(x)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.ndim == 2 and logits.shape[-1] == 1:
        return torch.sigmoid(logits[:, 0]).item()
    if logits.ndim == 0 or logits.numel() == 1:
        return torch.sigmoid(logits).item()

    if logits.ndim == 2 and logits.shape[-1] == 2:
        probs = torch.softmax(logits, dim=-1) 
        return probs[0, 1].item() 

    raise ValueError(f"Saída inesperada do modelo: shape={tuple(logits.shape)}")

POS_IS_YAWN = True 

def predict_label(sample_np: np.ndarray) -> Tuple[str, float]:
    """
    Classifica o vetor de entrada como 'Yawn' ou 'No Yawn' e retorna a confiança.

    Args:
        sample_np (np.ndarray): Vetor de características extraídas do rosto.

    Returns:
        Tuple[str, float]: Rótulo previsto e confiança.
    """

    p1 = predict_proba(sample_np)  
    if POS_IS_YAWN:
        label = "Yawn" if p1 >= 0.5 else "No Yawn"
        conf = p1 if label == "Yawn" else (1.0 - p1)
    else:
        label = "No Yawn" if p1 >= 0.5 else "Yawn"
        conf = p1 if label == "No Yawn" else (1.0 - p1)
    return label, float(conf)


def decidir_estado(frame: np.ndarray) -> Tuple[str, float]:
    """
    Decide o estado atual ('Yawn' ou 'No Yawn') a partir do frame da webcam.

    Args:
        frame (np.ndarray): Imagem do frame da webcam.

    Returns:
        Tuple[str, float]: Rótulo previsto e confiança.
    """
    
    return predict_label(centralizar_normalizar(embedar_mapeamento(frame)))

def main() -> None:
    """
    Função principal que inicializa a captura da webcam, processa os frames,
    detecta bocejos e emite alerta sonoro se necessário.
    """
    global t0
    state_list = deque(["No Yawn"]*5, maxlen=5)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

    if not cap.isOpened():
        print("Não foi possível acessar a câmera")
        return

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha ao capturar imagem")
                break
            t1 = time.perf_counter()
            if t1 - t0 > .4:
                t0 = t1
                estado = decidir_estado(frame)  
                print(f"Próximo estado: {estado}")
                state_list.append(estado[0])
            if state_list.count("Yawn") >= 4:
                play_wav()
            cv2.imshow("Camera", frame)
       
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break


    except KeyboardInterrupt:
        print("Encerrado pelo usuário.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()