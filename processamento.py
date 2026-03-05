import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import io, base64

# --- 1. Geração de Dados (Mackey-Glass simplificado para o browser) ---
def generate_data(N_samples=1500, tau=17):
    # Equação diferencial de atraso simples
    y = np.random.uniform(0.5, 1.5, tau)
    b, c = 0.1, 0.2
    for t in range(tau - 1, N_samples + tau - 1):
        dy = (c * y[t - tau]) / (1 + y[t - tau]**10) - b * y[t]
        y = np.append(y, y[t] + dy)
    data = y[tau:]
    return (data - np.mean(data)) / np.std(data)

# --- 2. A Classe Reservoir Computing (ESN) ---
class ESN:
    def __init__(self, n_reservoir=200, spectral_radius=0.9, alpha=0.3):
        self.N = n_reservoir
        self.sr = spectral_radius
        self.alpha = alpha
        
        # Pesos fixos aleatórios
        np.random.seed(42)
        self.W_in = np.random.uniform(-1, 1, (self.N, 1))
        W = np.random.uniform(-1, 1, (self.N, self.N))
        
        # Ajustando o raio espectral
        rhoW = max(abs(np.linalg.eigvals(W)))
        self.W = W * (self.sr / rhoW)
        self.readout = Ridge(alpha=1e-4)

    def fit(self, u, y):
        states = np.zeros((len(u), self.N))
        x = np.zeros((self.N, 1))
        
        # Coletando os estados do reservatório
        for t in range(len(u)):
            u_t = np.array([[u[t]]])
            # A equação mágica do seu notebook:
            x = (1 - self.alpha) * x + self.alpha * np.tanh(self.W_in @ u_t + self.W @ x)
            states[t] = x.flatten()
            
        # Treinando apenas a camada de saída (readout)
        self.readout.fit(states, y)
        self.last_state = x

    def forecast(self, last_y, steps):
        preds = []
        x = self.last_state
        curr_y = last_y
        
        # Feedback loop: a previsão vira a próxima entrada
        for t in range(steps):
            u_t = np.array([[curr_y]])
            x = (1 - self.alpha) * x + self.alpha * np.tanh(self.W_in @ u_t + self.W @ x)
            curr_y = self.readout.predict(x.T)[0]
            preds.append(curr_y)
