# Computação de Reservatório no Navegador (Web ESN)

Este repositório contém o código-fonte de uma aplicação web interativa que demonstra o conceito de **Reservoir Computing (Computação de Reservatório)** 
utilizando **Echo State Networks (ESNs)** para a previsão de séries temporais caóticas.

O grande diferencial deste projeto é que todo o processamento matemático, treinamento da rede neural e geração de gráficos são executados 
**nativamente no navegador do usuário**, sem a necessidade de um backend em Python ou de qualquer instalação local.

## Aplicação ao Vivo

Você pode testar a aplicação diretamente pelo link abaixo:

**[Acessar o Projeto (GitHub Pages)](https://samuel-sanches-br.github.io/)**

---

## Sobre o Projeto

As Echo State Networks (ESNs) são um tipo de Rede Neural Recorrente (RNN) onde a camada oculta (o "reservatório") possui pesos fixos e gerados aleatoriamente. 
Apenas a camada de saída (*readout*) é treinada, geralmente através de uma regressão linear simples (Ridge Regression). Isso torna o treinamento extremamente 
rápido e imune ao problema de desvanecimento do gradiente (*vanishing gradient*).

Para tornar essa experimentação acessível a qualquer pessoa, transformei o script original em Python numa página web interativa. 

### Principais Funcionalidades:
- **Execução Client-Side:** O código Python roda 100% no lado do cliente utilizando WebAssembly.
- **Visualização Integrada:** Geração dos gráficos de resultados (Série de dados e previsões) renderizados diretamente na interface.
- **Acessibilidade:** Perfeito para fins educacionais e demonstração rápida de conceitos de Machine Learning.

---

## Tecnologias Utilizadas

- **Front-end:** HTML5, CSS3
- **Execução Python no Web:** PyScript / WebAssembly (Pyodide)
- **Machine Learning & Matemática:** `numpy`, `scikit-learn`
- **Visualização de Dados:** `matplotlib`
