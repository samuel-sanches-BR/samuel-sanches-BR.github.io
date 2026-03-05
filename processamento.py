import math
import matplotlib.pyplot as plt
import numpy as np
import io, base64
from js import document

def calcular_e_plotar(valor_input):
    try:
        n = float(valor_input)
        if n < 0:
            return "Erro: O número deve ser positivo| "
            
        raiz = math.sqrt(n)
        
        # Lógica do Gráfico
        plt.clf()
        plt.figure(figsize=(5,3))
        x = np.linspace(0, n*1.2 if n > 0 else 10, 100)
        y = np.sqrt(x)
        
        plt.plot(x, y, color='#3498db')
        plt.scatter([n], [raiz], color='red')
        plt.grid(True)
        
        # Converte para Base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        img_str = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('UTF-8')
        
        return f"A raiz quadrada de {n} é {raiz:.4f}|{img_str}"
    except Exception as e:
        return f"Erro no processamento: {str(e)}| "
