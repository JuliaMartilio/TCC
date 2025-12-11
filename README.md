## Sistema de Triagem de Assimetria Postural (TCC)

Este projeto implementa um sistema de triagem automatizado para desvios posturais. O sistema utiliza a Visão Computacional, especificamente **Marcadores ArUco**, para medir o desnível em centímetros nas regiões dos ombros, escápulas e pelve. As métricas obtidas são então inseridas em **três modelos de Decision Tree (IA)**, que classificam a assimetria em categorias clínicas de forma segmentada. O objetivo é fornecer uma ferramenta objetiva, rápida e de baixo custo para auxiliar na triagem inicial de pacientes com possíveis alterações posturais relacionadas a escoliose.

O projeto aplica ainda **Marcadores digitais Mediapipe** para medir o assimetria em centímetros apenas na região dos ombros e pelve, de modo a comparar os resultados obtidos nesta aplicação com os dados coletados ao utilizar os Marcadores ArUco, tendo como objetivo comparar a acurácia do uso de marcadores físicos com o uso dos marcadores digitais.
