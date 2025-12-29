from datetime import datetime, timedelta

LOCALE = 'pt_BR'
NUM_STORES = 100
NUM_PRODUCTS = 200
HISTORY_YEARS = 3
END_DATE = datetime(2024, 12, 31)
START_DATE = END_DATE - timedelta(days=365 * HISTORY_YEARS)
OUTPUT_DIR = "/opt/data/raw"

REGIONAL_GEO_MAP = {
    'Regional_Sul': {
        'RS': ['Porto Alegre', 'Caxias do Sul', 'Pelotas', 'Santa Maria'],
        'SC': ['Florianópolis', 'Joinville', 'Balneário Camboriú'],
        'PR': ['Curitiba', 'Londrina', 'Maringá']
    },
    'Regional_SP': {
        'SP': ['São Paulo', 'Campinas', 'Santos', 'Ribeirão Preto', 'Sorocaba']
    },
    'Regional_Rio_Minas': {
        'RJ': ['Rio de Janeiro', 'Niterói', 'Petrópolis'],
        'MG': ['Belo Horizonte', 'Uberlândia', 'Ouro Preto'],
        'ES': ['Vitória', 'Vila Velha']
    },
    'Regional_Nordeste': {
        'BA': ['Salvador', 'Feira de Santana'],
        'PE': ['Recife', 'Caruaru'],
        'CE': ['Fortaleza'],
        'RN': ['Natal']
    },
    'Regional_Centro_Norte': {
        'DF': ['Brasília'],
        'GO': ['Goiânia'],
        'AM': ['Manaus'],
        'PA': ['Belém']
    }
}
