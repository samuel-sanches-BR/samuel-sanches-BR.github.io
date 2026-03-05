import math
import matplotlib.pyplot as plt
import numpy as np
import io, base64
from js import document

def calcular_e_plotar(valor_bruto):
    try:
        # Tenta converter a entrada do usuário para número
        # O tratamento de erro começa aqui
        valor_limpo = valor_bruto.replace(',', '.') # Troca vírgula por ponto
        n = float(valor_limpo)
        
        if n < 0:
            return "Erro: Não existe raiz quadrada real de número negativo.| "
            
        raiz = math.sqrt(n)
        
        # Criação do Gráfico
        plt.clf()
        plt.figure(figsize=(5,3))
        
        # Define o alcance do gráfico baseado no número escolhido
        limite_x = max(n * 1.3, 5)
        x = np.linspace(0, limite_x, 100)
        y = np.sqrt(x)
        
        plt.plot(x, y, color='#3498db', linewidth=2)
        plt.scatter([n], [raiz], color='#e74c3c', s=50) # Ponto exato
        plt.title(f"Função Raiz Quadrada (n={n})")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Converte para base64 para o HTML exibir
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_str = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('UTF-8')
        
        return f"A raiz de {n} é aproximadamente {raiz:.4f}|{img_str}"

    except ValueError:
        return "Erro: Por favor, digite um número válido (ex: 144 ou 12.5).| "
    except Exception as e:
        return f"Erro inesperado: {str(e)}| "
