import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import io, base64

plt.rcParams.update({
    'figure.facecolor': '#111827',
    'axes.facecolor':   '#0a0e1a',
    'axes.edgecolor':   '#1e2d45',
    'axes.labelcolor':  '#94a3b8',
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'text.color':       '#e2e8f0',
    'grid.color':       '#1e2d45',
    'grid.linewidth':   0.6,
    'axes.grid':        True,
    'legend.facecolor': '#111827',
    'legend.edgecolor': '#1e2d45',
    'legend.fontsize':  8,
})

# ── Geradores de dados ─────────────────────────────────────────────────────────

def mackey_glass(n_samples=1500, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    history_len = max(tau, 50)
    history = 0.5 * np.ones(history_len)
    x = list(history)
    for _ in range(n_samples):
        x_now = x[-1]
        x_tau = x[-tau] if len(x) >= tau else 0.5
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x_now
        x.append(x_now + dt * dx)
    return np.array(x[history_len:])


def lorenz(n_samples=1500, sigma=10.0, rho=28.0, beta=8/3, dt=0.01):
    burnin = 1000
    total  = n_samples + burnin
    xs = np.empty(total); ys = np.empty(total); zs = np.empty(total)
    xs[0], ys[0], zs[0] = 0.1, 0.0, 0.0
    for i in range(total - 1):
        xs[i+1] = xs[i] + dt * sigma * (ys[i] - xs[i])
        ys[i+1] = ys[i] + dt * (xs[i] * (rho - zs[i]) - ys[i])
        zs[i+1] = zs[i] + dt * (xs[i] * ys[i] - beta * zs[i])
    return xs[burnin:]


def get_data(equation, n_samples=1500):
    raw = lorenz(n_samples=n_samples) if equation == 'lorenz' else mackey_glass(n_samples=n_samples)
    return (raw - raw.min()) / (raw.max() - raw.min())


# ── Preview dos dados ──────────────────────────────────────────────────────────

def preview_data(equation='mackey_glass'):
    try:
        data    = get_data(equation)
        n_train = 1000
        t       = np.arange(len(data))

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

        # Série com split treino/teste
        axes[0].plot(t[:n_train], data[:n_train],
                     color='#00d4aa', linewidth=0.9, label=f'Treino  ({n_train} pts)')
        axes[0].plot(t[n_train:], data[n_train:],
                     color='#ff6b35', linewidth=0.9, label=f'Teste   ({len(data)-n_train} pts)')
        axes[0].axvline(n_train, color='#475569', linestyle='--', linewidth=1.0)
        axes[0].set_title('Série Temporal', fontsize=10)
        axes[0].set_xlabel('Passo de tempo')
        axes[0].set_ylabel('x(t)  [normalizado]')
        axes[0].legend(loc='upper right')

        # Retrato de fase
        tau_phase = 17 if equation == 'mackey_glass' else 10
        axes[1].plot(data[:-tau_phase], data[tau_phase:],
                     '.', color='#94a3b8', alpha=0.2, markersize=1.2)
        axes[1].set_title(f'Atrator  —  x(t) vs x(t+{tau_phase})', fontsize=10)
        axes[1].set_xlabel(f'x(t)')
        axes[1].set_ylabel(f'x(t+{tau_phase})')

        plt.tight_layout(pad=1.2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=110)
        img = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('UTF-8')
        plt.close(fig)
        return img
    except Exception as e:
        return f'ERRO|{str(e)}'


# ── ESN ────────────────────────────────────────────────────────────────────────

class EchoStateNetwork:
    def __init__(self, n_reservoir=500, spectral_radius=0.95,
                 sparsity=0.1, input_scaling=1.0, leaking_rate=1.0,
                 n_inputs=1, n_outputs=1, washout=100, ridge_alpha=1e-6):
        self.N = n_reservoir; self.sr = spectral_radius
        self.sparsity = sparsity; self.input_scaling = input_scaling
        self.alpha = leaking_rate; self.n_in = n_inputs
        self.n_out = n_outputs; self.washout = washout
        self.ridge_alpha = ridge_alpha
        self._build_reservoir()

    def _build_reservoir(self):
        np.random.seed(42)
        self.W_in = (np.random.rand(self.N, self.n_in) - 0.5) * self.input_scaling
        W = np.random.rand(self.N, self.N) - 0.5
        W[np.random.rand(self.N, self.N) > self.sparsity] = 0
        self.W = W * (self.sr / np.max(np.abs(np.linalg.eigvals(W))))

    def _run_reservoir(self, u):
        T = len(u); states = np.zeros((T, self.N)); x = np.zeros(self.N)
        for t in range(T):
            x = (1-self.alpha)*x + self.alpha*np.tanh(self.W_in @ u[t] + self.W @ x)
            states[t] = x
        return states

    def fit(self, u_train, y_train):
        s  = self._run_reservoir(u_train)
        sf = s[self.washout:]; yf = y_train[self.washout:]
        ext = np.hstack([sf, u_train[self.washout:]])
        self.readout = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.readout.fit(ext, yf)
        return self

    def predict(self, u_test):
        s = self._run_reservoir(u_test)
        return self.readout.predict(np.hstack([s, u_test]))


# ── Forecast autoregressivo ────────────────────────────────────────────────────

def forecast(esn, u_train, u_test, last_val, n_steps=100):
    x   = esn._run_reservoir(np.vstack([u_train, u_test]))[-1]
    cur = np.array([[last_val]]); preds = []
    for _ in range(n_steps):
        x   = (1-esn.alpha)*x + esn.alpha*np.tanh(esn.W_in @ cur[0] + esn.W @ x)
        nv  = esn.readout.predict(np.hstack([x, cur[0]]).reshape(1,-1))[0]
        preds.append(nv); cur = np.array([[nv]])
    return np.array(preds)


# ── Função principal ───────────────────────────────────────────────────────────

def run_esn_app(n_res_str, spec_rad_str, leak_rate_str,
                n_future_str="100", equation="mackey_glass"):
    try:
        n_res = int(n_res_str); spec_rad = float(spec_rad_str)
        leak_rate = float(leak_rate_str); n_future = int(n_future_str)

        data = get_data(equation)
        u = data[:-1].reshape(-1, 1); y = data[1:]
        n_train = 1000
        u_tr, u_te = u[:n_train], u[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]

        esn = EchoStateNetwork(n_reservoir=n_res, spectral_radius=spec_rad,
                               leaking_rate=leak_rate, sparsity=0.1,
                               input_scaling=0.1, washout=100, ridge_alpha=1e-4)
        esn.fit(u_tr, y_tr)

        y_pred = esn.predict(u_te)
        nrmse  = np.sqrt(mean_squared_error(y_te, y_pred)) / np.std(y_te)
        fut    = forecast(esn, u_tr, u_te, u_te[-1, 0], n_steps=n_future)

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        t = np.arange(len(y_te))
        eq_label = 'Mackey-Glass' if equation == 'mackey_glass' else 'Lorenz'

        axes[0].plot(t[:200], y_te[:200], color='#00d4aa', linewidth=1.8, label='Real')
        axes[0].plot(t[:200], y_pred[:200], color='#ff6b35', linewidth=1.8,
                     linestyle='--', label='Previsão ESN')
        axes[0].set_title(f'{eq_label}  ·  One-Step-Ahead  (NRMSE: {nrmse:.4f})  |  Neurônios: {n_res}', fontsize=10)
        axes[0].set_ylabel('x(t+1)'); axes[0].legend()

        err = np.abs(y_te - y_pred)
        axes[1].fill_between(t[:200], err[:200], alpha=0.35, color='#e74c3c')
        axes[1].plot(t[:200], err[:200], color='#e74c3c', linewidth=0.9)
        axes[1].axhline(err[:200].mean(), color='#c0392b', linestyle='--',
                        label=f'Erro médio = {err[:200].mean():.4f}')
        axes[1].set_title('Erro Absoluto (primeiros 200 passos)', fontsize=10)
        axes[1].set_ylabel('|erro|'); axes[1].legend()

        t_past = np.arange(-100, 0); t_fut = np.arange(0, n_future)
        axes[2].plot(t_past, y_te[-100:], color='#00d4aa', linewidth=1.8, label='Dados reais')
        axes[2].plot(t_fut, fut, color='#ff6b35', linewidth=1.8,
                     linestyle='--', label=f'Forecast ({n_future} passos)')
        axes[2].axvline(0, color='#475569', linestyle=':', linewidth=1.0, label='Início do forecast')
        axes[2].set_title(f'Forecast Autoregressivo — {n_future} passos no futuro', fontsize=10)
        axes[2].set_xlabel('Passos (0 = último ponto real)')
        axes[2].set_ylabel('x(t)'); axes[2].legend()

        plt.tight_layout(pad=1.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        img = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('UTF-8')
        plt.close(fig)

        return f"Modelo treinado com sucesso! NRMSE: {nrmse:.4f}|{img}"

    except Exception as e:
        return f"Erro ao rodar ESN: {str(e)}| "
