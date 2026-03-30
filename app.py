import streamlit as st 
import numpy as np 
import pandas as pd 
import joblib

#import torch

#torch_load_original = torch.load

#def torch_load_cpu(*args, **kwargs):
#    kwargs['map_location'] = torch.device('cpu')
#    return torch_load_original(*args, **kwargs)

#torch.load = torch_load_cpu

modelo=joblib.load('./modelo_treinado_cpu.lib')
processador=joblib.load('./preprocessador_treinado.lib')

st.set_page_config(layout="wide")


st.image('./logo_telessaude.jpeg')
st.title('Preditor de limitações funcionais relacionadas a diabetes.')

#col1, col2 = st.columns(2)

#with col1:
with st.expander('Sobre'):
        st.text("""
            O aplicativo tem o objetivo de auxiliar profissionais de saúde a tomarem decisões sobre o tratamento de pacientes que convivem com diabetes, de acordo com a probabilidade de desenvolverem limitações funcionais.""",
                text_alignment='justify' )

#with col2:
with st.expander('Como usar'):
        st.write("""
            - O profissional deve preencher o questionário ao lado com as respostas dos pacientes.
            - O aplicativo irá devolver a probabilidade do paciente desenvolver uma limitação funcional.
            - O limiar sugerido para iniciar intervenções preventivas é entre **35-60%**.
        """)
modelo=joblib.load('./modelo_treinado.lib')
processador=joblib.load('./preprocessador_treinado.lib')

with st.sidebar:
    tempo_tv= {1:'Menos de uma hora',
           2:'De uma hora a menos de duas horas',
           3:'De duas horas a menos de três horas',
           4:'De três horas a menos de seis horas',
           5:'Seis horas ou mais',
           6:'Não consumo'}

    resposta_binaria={1:"Sim",2:'Não'}
    
    etnia={
            1:'Branca',
            2:'Preta',
            3:'Amarela',
            4:'Parda', 
            5:'Indígena'}

    st.header('Questionário:')
    
    idade = st.slider('Idade do morador na data de referência', 0, 120, 45)
    cor_raca = st.selectbox('Cor ou raça',
                            options=[*etnia],
                            format_func= lambda x: etnia.get(x) )
    renda = st.selectbox(
        'Faixa de rendimento domiciliar per capita',
        options=[1, 2, 3, 4, 5, 6, 7],
        format_func=lambda x: {
        1: 'Até ¼ salário mínimo',
        2: 'Mais de ¼ até ½ salário mínimo',
        3: 'Mais de ½ até 1 salário mínimo',
        4: 'Mais de 1 até 2 salários mínimos',
        5: 'Mais de 2 até 3 salários mínimos',
        6: 'Mais de 3 até 5 salários mínimos',
        7: 'Mais de 5 salários mínimos',
        }[x]
    )

   

    feijao = st.slider('Dias/semana que come feijão', 0, 7, 3)
    verdura_legume = st.slider('Dias/semana que come verdura ou legume', 0, 7, 3)
    carne_vermelha = st.slider('Dias/semana que come carne vermelha', 0, 7, 3)
    frango = st.slider('Dias/semana que come frango/galinha', 0, 7, 3)
    peixe = st.slider('Dias/semana que come peixe', 0, 7, 2)
    suco_natural = st.slider('Dias/semana que toma suco de fruta natural', 0, 7, 2)
    frutas = st.slider('Dias/semana que come frutas', 0, 7, 3)
    suco_caixinha = st.slider('Dias/semana que toma suco de caixinha/refresco em pó', 0, 7, 1)
    refrigerante = st.slider('Dias/semana que toma refrigerante', 0, 7, 1)
    doces = st.slider('Dias/semana que come alimentos doces', 0, 7, 2)
    leite = st.slider('Dias/semana que toma leite', 0, 7, 3)

    horas_tv = st.selectbox('Horas/dia assistindo televisão',
                            options=[*tempo_tv],
                            format_func=lambda x: tempo_tv[x]
                            )
    horas_tela = st.selectbox('Horas/dia usando computador/tablet/celular para lazer',
                              options=[1,2,3,4,5,6],
                            format_func=lambda x: tempo_tv[x]
    )

    insulina = st.selectbox('Médico já receitou insulina para controlar o Diabetes?', 
                            options=[1,2],
                            format_func= lambda x: resposta_binaria[x]
                            )
    internacao = st.selectbox('Ficou internado em hospital por 24h+ nos últimos 12 meses?',
                                options=[1,2],
                                format_func= lambda x: resposta_binaria[x]
)
    consultas_medico = st.slider('Consultas ao médico nos últimos 12 meses', 0, 30, 2)
    duracao = st.slider('Tempo de diagnóstico de diabetes (anos)', 0, 50, 5)


data = pd.DataFrame({
    'C008':   [idade],
    'C009':   [cor_raca],
    'VDF004': [renda],
    'P006':   [feijao],
    'P00901': [verdura_legume],
    'P01101': [carne_vermelha],
    'P013':   [frango],
    'P015':   [peixe],
    'P01601': [suco_natural],
    'P018':   [frutas],
    'P02001': [suco_caixinha],
    'P02002': [refrigerante],
    'P02501': [doces],
    'P023':   [leite],
    'P04501': [horas_tv],
    'P04502': [horas_tela],
    'Q03802': [insulina],
    'J037':   [internacao],
    'J012':   [consultas_medico],
    'duracao':[duracao],
})

# Pré-processamento e predição
# st.subheader('Resultado da predição')
if st.button('Prever'):
    with st.spinner("Calculando a probabilidade... "):
        data_processada = processador.transform(data)
        #predicao = modelo.predict(data_processada)
        probabilidade = modelo.predict_proba(data_processada)
    
    st.subheader(f'A probabilidade do paciente ter uma limitação é de {probabilidade[0][1]:.1%}')
    #if predicao[0] == 1:
    #    st.error(f'⚠️ Paciente com risco de limitação funcional  (probabilidade: {probabilidade[0][1]:.1%})')
    #else:
    #    st.success(f'✅ Paciente sem risco de limitação funcional (probabilidade: {probabilidade[0][0]:.1%})')

