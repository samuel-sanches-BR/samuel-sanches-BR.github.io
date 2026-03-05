# samuel-sanches-BR.github.io
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Meu Site com Pyodide</title>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js"></script>
</head>
<body>
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
