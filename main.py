import pandas as pd
import os.path
import requests


url = "https://zenodo.org/record/7792203/files/proposicoes_1988_2022.csv"
directory = './dataset/'

def create_chat_prompt(text):
    return [
        {"role":"system","content":
                      'Considere as categorias: "Administração Pública", "Arte, Cultura e Religião","Comunicações", "Esporte e Lazer", "Economia", "Cidades e Desenvolvimento Urbano","Direito Civil e Processual Civil", "Direito Penal e Processual Penal", "Direitos Humanos e Minorias","Educação", "Meio Ambiente e Desenvolvimento Sustentável","Estrutura Fundiária", "Previdência e Assistência Social","Processo Legislativo e Atuação Parlamentar","Energia, Recursos Hídricos e Minerais","Relações Internacionais e Comércio Exterior", "Saúde","Defesa e Segurança", "Trabalho e Emprego", "Turismo","Viação, Transporte e Mobilidade","Ciência, Tecnologia e Inovação","Agricultura, Pecuária, Pesca e Extrativismo","Indústria, Comércio e Serviços", "Direito e Defesa do Consumidor","Direito Constitucional", "Finanças Públicas e Orçamento","Homenagens e Datas Comemorativas","Política, Partidos e Eleições", "Direito e Justiça","Ciências Sociais e Humanas".'
                }, 
        {"role": "user", "content": 'classifique isto: '+text}
    ]

def join_function(value):
    return [str(i) for i in value]

r = requests.get(url, allow_redirects=True)
if url.find('/'):
    filename = url.rsplit('/', 1)[1]
    file_path = os.path.join(directory, filename)
    open(file_path, 'wb').write(r.content)
    print('obtido ' + url.rsplit('/', 1)[1])

proposicoes = pd.read_csv('./dataset/proposicoes_1988_2022.csv', dtype="object")

ementas = pd.DataFrame(proposicoes[proposicoes['siglaTipo'].isin(['PL','PDL','MPV'])], columns=['ementa','tema'])
ementas_limpas = ementas.dropna()
ementas_temas = ementas_limpas.groupby(['ementa'])['tema'].apply(join_function).reset_index()
amostra = ementas_temas.sample(1000)
amostra["input"], amostra["ideal"] = amostra["ementa"].apply(create_chat_prompt), amostra["tema"]
amostra_pronta = amostra[["input", "ideal"]]
amostra_pronta.to_json("samples.jsonl", orient="records", lines=True)

eval_yaml = """
brazilian_laws:
  id: brazilian_laws.test.v1
  metrics: [accuracy]
brazilian_laws.test.v1:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: brazilian_laws/samples.jsonl
    
brazilian_laws:
  id: brazilian_laws.test.v1
  metrics: [f1_score]
brazilian_laws.test.v1:
  class: evals.elsuite.basic.fuzzy_match:FuzzyMatch
  description: Example eval that uses fuzzy matching to score completions.
  args:
    samples_jsonl: brazilian_laws/samples.jsonl

brazilian_laws:
  id: brazilian_laws.test.v1
  metrics: [accuracy]
brazilian_laws.test.v1:
  class: evals.elsuite.basic.includes:Includes
  description: Example eval that uses fuzzy matching to score completions.
  args:
    samples_jsonl: brazilian_laws/samples.jsonl
""".strip()
with open("brazilian_laws.yaml", "w") as f:
    f.write(eval_yaml)

print('Concluído')
