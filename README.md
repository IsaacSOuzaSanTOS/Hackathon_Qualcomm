# Hackathon_Qualcomm

## Guardians

## Membros
- Antonio Iranilson Honorato dos Santos Junior - [iranhonorato](https://github.com/iranhonorato)
- Matheus Henrique Pereira Borba - [mthperera](https://github.com/mthperera)
- Gustavo Santana Silva - [YouCanCallMeGus](https://github.com/YouCanCallMeGus)
- Felipe Serra Silva - [Felipe-Serra-Silva](https://github.com/Felipe-Serra-Silva)
- Isaac Souza Santos - [IsaacSOuzaSanTOS](https://github.com/IsaacSOuzaSanTOS)

## Descrição do Projeto

Este projeto é um sistema de alerta para motoristas sonolentos que utiliza Edge AI para analisar em tempo real o rosto do condutor e detectar sinais de sonolência. A inteligência artificial identifica expressões faciais, movimentos dos olhos e padrões que indicam fadiga, gerando alertas imediatos para prevenir acidentes.

Toda a operação é feita 100% localmente, sem depender de sistemas de nuvem. O banco de dados e o processamento de informações permanecem na própria máquina, garantindo privacidade, segurança e baixa latência, mesmo em regiões sem conectividade.

O objetivo central é reduzir acidentes causados pela sonolência ao volante, oferecendo uma solução confiável e prática que combina visão computacional avançada e processamento embarcado, totalmente independente da internet.

## Tecnologias

O sistema foi desenvolvido com foco em **Edge AI**, utilizando tecnologias que permitem **processamento local**, **visão computacional** e **interação em tempo real com o usuário**. A arquitetura do projeto é dividida entre **backend** e **frontend**, garantindo eficiência, segurança e experiência intuitiva.

### Backend

O backend é responsável pelo **processamento de dados, inferência da IA e armazenamento local**. Todas as operações acontecem na própria máquina, sem uso de nuvem, garantindo **privacidade e baixa latência**.

- **OpenCV (`cv2`)**  
  Captura de vídeo em tempo real e processamento de imagens. Realiza operações como conversão de cores, detecção de rosto e pré-processamento de frames para a IA.

- **MediaPipe (`mediapipe`)**  
  Rastreamento facial e extração de landmarks (pontos de referência do rosto). Permite à IA identificar expressões faciais e sinais de sonolência, como fechamento dos olhos ou bocejos.

- **PyTorch (`torch`)**  
  Treinamento e inferência de modelos de detecção de sonolência. Executa o processamento de dados localmente e fornece predições em tempo real para alertar o motorista.

- **Scikit-Learn (`sklearn`)**  
  Utilizada para pré-processamento de dados, normalização e avaliação do desempenho do modelo, garantindo que a IA funcione de forma robusta e precisa.

### Frontend

O frontend é uma **interface de demonstração**, usada para visualizar como o backend processa os dados e gera alertas de sonolência. Ele permite acompanhar em tempo real o funcionamento da IA, mas não é necessário para o processamento principal, que ocorre totalmente no backend.

- **Next.js**  
  Framework utilizado para construir a interface de demonstração de forma rápida e interativa. Mostra alertas visuais e estatísticas sobre a detecção de sonolência, permitindo que o usuário veja o sistema em ação sem interferir na lógica do backend.

### Observação

Todo o **processamento e armazenamento de dados** ocorre **100% localmente**, sem uso de nuvem. Isso garante:
- Privacidade dos dados do motorista  
- Operação contínua mesmo em áreas sem internet  
- Baixa latência na detecção de sonolência e envio de alertas  

## Benefícios Esperados

Nosso sistema de alerta para motoristas sonolentos traz diversos benefícios, tornando-o uma solução inovadora e prática:

- **Custo reduzido e acessibilidade para todos**  
  Desenvolvido com tecnologias de código aberto e utilizando processamento local, o sistema evita custos com nuvem e infraestrutura complexa, tornando-se financeiramente viável para diferentes perfis de usuários.

- **Portabilidade completa e uso em qualquer veículo**
  O software roda totalmente em uma máquina local, sem depender de conectividade com a internet. Isso permite que seja usado em diferentes veículos, cidades ou regiões remotas, oferecendo flexibilidade e mobilidade.

- **Impacto direto na segurança de motoristas e passageiros** 
  Ao prevenir acidentes causados por fadiga, o sistema impacta diretamente a segurança no trânsito, protegendo motoristas, passageiros e pedestres, ampliando seu efeito social positivo.

- **Máximo aproveitamento do hardware do processador**
  Aproveita ao máximo as capacidades do Edge AI e do hardware local, realizando análise de vídeo em tempo real, inferência do modelo de IA e envio de alertas imediatos, sem depender de servidores externos.

- **Privacidade e segurança dos dados**  
  Todo o processamento e armazenamento dos dados ocorre localmente, garantindo que informações sensíveis, como imagens do rosto do motorista, permaneçam privadas e seguras.

- **Detecção em tempo real com alta precisão**  
  A inteligência artificial analisa expressões faciais, movimentos dos olhos e outros sinais de sonolência em tempo real, permitindo alertas rápidos e eficazes, essenciais para reduzir acidentes.

- **Alta confiabilidade e baixa latência**  
  Por operar totalmente de forma local, sem depender de nuvem ou internet, o sistema envia alertas instantâneos com precisão e funciona de maneira confiável em qualquer situação.

## Link de Demonstração



## Instalação e Execução
