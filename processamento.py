import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import io, base64

# 1. Geração de dados
def mackey_glass(n_samples=1500, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    history_len = max(tau, 50)
    history = 0.5 * np.ones(history_len)
    x = list(history)
    for i in range(n_samples):
        x_now = x[-1]
        x_tau = x[-tau] if len(x) >= tau else 0.5
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x_now
        x.append(x_now + dt * dx)
    return np.array(x[history_len:])


# 2. Classe ESN
class EchoStateNetwork:
    def __init__(self, n_reservoir=500, spectral_radius=0.95,
                 sparsity=0.1, input_scaling=1.0, leaking_rate=1.0,
                 n_inputs=1, n_outputs=1, washout=100, ridge_alpha=1e-6):
        self.N = n_reservoir
        self.sr = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.alpha = leaking_rate
        self.n_in = n_inputs
        self.n_out = n_outputs
        self.washout = washout
        self.ridge_alpha = ridge_alpha
        self._build_reservoir()

    def _build_reservoir(self):
        np.random.seed(42)
        self.W_in = (np.random.rand(self.N, self.n_in) - 0.5) * self.input_scaling
        W = np.random.rand(self.N, self.N) - 0.5
        W[np.random.rand(self.N, self.N) > self.sparsity] = 0
        eigenvalues = np.linalg.eigvals(W)
        current_sr = np.max(np.abs(eigenvalues))
        self.W = W * (self.sr / current_sr)

    def _run_reservoir(self, u):
        T = len(u)
        states = np.zeros((T, self.N))
        x = np.zeros(self.N)
        for t in range(T):
            x = ((1 - self.alpha) * x
                 + self.alpha * np.tanh(self.W_in @ u[t] + self.W @ x))
            states[t] = x
        return states

    def fit(self, u_train, y_train):
        states = self._run_reservoir(u_train)
        states_fit = states[self.washout:]
        y_fit = y_train[self.washout:]
        extended_states = np.hstack([states_fit, u_train[self.washout:]])
        self.readout = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.readout.fit(extended_states, y_fit)
        return self

    def predict(self, u_test):
        states = self._run_reservoir(u_test)
        extended_states = np.hstack([states, u_test])
        return self.readout.predict(extended_states)


# 3. Função de forecast autoregressivo (do notebook, célula 16)
def forecast(esn, u_train, u_test, last_known_input, n_steps=100):
    """
    Gera n_steps previsões para o futuro de forma autoregressiva.
    O modelo alimenta suas próprias previsões de volta como input.
    """
    forecasts = []

    # Inicializa o estado do reservatório com todos os dados reais vistos
    full_u = np.vstack([u_train, u_test])
    states = esn._run_reservoir(full_u)
    x = states[-1]  # último estado do reservatório

    current_input = np.array([[last_known_input]])

    for step in range(n_steps):
        # Atualiza o reservatório
        x = ((1 - esn.alpha) * x
             + esn.alpha * np.tanh(esn.W_in @ current_input[0] + esn.W @ x))

        # Monta estado estendido e gera próxima previsão
        extended = np.hstack([x, current_input[0]]).reshape(1, -1)
        next_val = esn.readout.predict(extended)[0]
        forecasts.append(next_val)

        # Realimenta a previsão como próximo input
        current_input = np.array([[next_val]])

    return np.array(forecasts)


# 4. Função principal chamada pelo JavaScript
def run_esn_app(n_res_str, spec_rad_str, leak_rate_str, n_future_str="100"):
    try:
        n_res     = int(n_res_str)
        spec_rad  = float(spec_rad_str)
        leak_rate = float(leak_rate_str)
        n_future  = int(n_future_str)

        # Gera e normaliza dados
        data = mackey_glass(n_samples=1500)
        data = (data - data.min()) / (data.max() - data.min())

        u = data[:-1].reshape(-1, 1)
        y = data[1:]

        n_train = 1000
        u_train, u_test = u[:n_train], u[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Treina o modelo
        esn = EchoStateNetwork(
            n_reservoir=n_res,
            spectral_radius=spec_rad,
            leaking_rate=leak_rate,
            sparsity=0.1,
            input_scaling=0.1,
            washout=100,
            ridge_alpha=1e-4
        )
        esn.fit(u_train, y_train)

        # Previsão one-step-ahead no conjunto de teste
        y_pred = esn.predict(u_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nrmse = test_rmse / np.std(y_test)

        # Previsão autoregressiva para o futuro
        last_real_value = u_test[-1, 0]
        future_preds = forecast(esn, u_train, u_test, last_real_value, n_steps=n_future)

        # ── Geração dos gráficos (3 subplots) ──────────────────────────────────
        plt.clf()
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        t = np.arange(len(y_test))

        # Plot 1: Verdadeiro vs Predito (one-step-ahead)
        axes[0].plot(t[:200], y_test[:200], label='Real', color='steelblue', linewidth=2)
        axes[0].plot(t[:200], y_pred[:200], label='Previsão ESN', color='orangered',
                     linewidth=2, linestyle='--')
        axes[0].set_title(f'Previsão One-Step-Ahead (NRMSE: {nrmse:.4f}) | Neurônios: {n_res}', fontsize=11)
        axes[0].set_ylabel('x(t+1)')
        axes[0].legend()

        # Plot 2: Erro absoluto
        error = np.abs(y_test - y_pred)
        axes[1].fill_between(t[:200], error[:200], alpha=0.4, color='crimson')
        axes[1].plot(t[:200], error[:200], color='crimson', linewidth=1)
        axes[1].axhline(error[:200].mean(), color='darkred', linestyle='--',
                        label=f'Erro médio = {error[:200].mean():.4f}')
        axes[1].set_title('Erro Absoluto (Primeiros 200 passos)', fontsize=11)
        axes[1].set_ylabel('|erro|')
        axes[1].legend()

        # Plot 3: Forecast autoregressivo futuro
        context = 100
        t_past   = np.arange(-context, 0)
        t_future = np.arange(0, n_future)

        axes[2].plot(t_past, y_test[-context:], color='steelblue', linewidth=1.8, label='Dados reais')
        axes[2].plot(t_future, future_preds, color='orangered', linewidth=1.8,
                     linestyle='--', label=f'Forecast ({n_future} passos)')
        axes[2].axvline(0, color='gray', linestyle=':', linewidth=1.2, label='Início da previsão')
        axes[2].set_title(f'Forecast Autoregressivo — {n_future} Passos no Futuro', fontsize=11)
        axes[2].set_xlabel('Passos (0 = último ponto conhecido)')
        axes[2].set_ylabel('x(t)')
        axes[2].legend()

        plt.tight_layout()

        # Converte para imagem base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        img_str = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('UTF-8')
        plt.close(fig)

        msg = f"Modelo treinado com sucesso! NRMSE: {nrmse:.4f}"
        return f"{msg}|{img_str}"

    except Exception as e:
        return f"Erro ao rodar ESN: {str(e)}| "
