import math
import matplotlib.pyplot as plt
import numpy as np
import io, base64

def calcular_e_plotar(entrada):
    try:
        # Garante que qualquer vírgula remanescente vire ponto
        num_str = str(entrada).replace(',', '.')
        n = float(num_str)
        
        if n < 0:
            return "Erro: O número deve ser positivo para raiz real.| "
            
        raiz = math.sqrt(n)
        
        # Limpa e configura o gráfico
        plt.clf()
        fig = plt.figure(figsize=(6, 4))
        
        limite = max(n * 1.5, 10)
        x = np.linspace(0, limite, 100)
        y = np.sqrt(x)
        
        plt.plot(x, y, color='#3498db', label='f(x) = √x', linewidth=2)
        plt.scatter([n], [raiz], color='#e74c3c', s=100, zorder=5)
        plt.title(f"Resultado: √{n} ≈ {raiz:.2f}", fontsize=12)
        plt.xlabel("X")
        plt.ylabel("√X")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Converte para base64 para o download e exibição
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        img_str = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('UTF-8')
        plt.close(fig)
        
        return f"Sucesso! A raiz quadrada de {n} é {raiz:.4f}|{img_str}"

    except ValueError:
        return "Erro: Digite um número válido.| "
    except Exception as e:
        return f"Erro inesperado: {str(e)}| "
