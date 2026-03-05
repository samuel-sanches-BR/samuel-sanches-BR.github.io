<html lang="pt-br">

    <h1>Python no Navegador!</h1>
    <p>Resultado do Python: <span id="output">Carregando...</span></p>

    <script type="text/javascript">
      async function main() {
        // Inicializa o Pyodide
        let pyodide = await loadPyodide();
        
        // Exemplo: Rodando um código Python simples
        let resultado = await pyodide.runPythonAsync(`
            import math
            x = math.sqrt(144)
            f"A raiz quadrada de 144 é {x}"
        `);

        // Exibe o resultado no HTML
        document.getElementById("output").innerText = resultado;
      }
      
      main();
    </script>
</body>
</html>
<div id="loading">Carregando Python e Bibliotecas (isso pode levar 10s)...</div>
<div id="plot-container"></div>

<script src="https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js"></script>
<script type="text/javascript">
  async function runPlot() {
    // 1. Inicializa o Pyodide
    let pyodide = await loadPyodide();
    
    // 2. Carrega o Matplotlib e dependências
    await pyodide.loadPackage(["matplotlib", "numpy"]);
    
    // Remove a mensagem de carregamento
    document.getElementById("loading").style.display = "none";

    // 3. Código Python para gerar o gráfico
    await pyodide.runPythonAsync(`
        import matplotlib.pyplot as plt
        import numpy as np
        from js import document

        # Criando dados para o gráfico
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        # Configurando o gráfico
        plt.figure(figsize=(6,4))
        plt.plot(x, y, label='Seno(x)', color='blue')
        plt.title('Gráfico Gerado com Python no Navegador')
        plt.grid(True)
        plt.legend()

        # Função especial do Pyodide para renderizar no HTML
        def plot_to_div(div_id):
            import io
            import base64
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('UTF-8')
            
            img_tag = document.createElement('img')
            img_tag.src = img_str
            document.getElementById(div_id).appendChild(img_tag)

        plot_to_div('plot-container')
    `);
  }

  runPlot();
</script>

<style>
  #loading { font-family: sans-serif; color: #666; padding: 20px; }
  #plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
</style>
