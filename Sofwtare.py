import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Funções de Cálculo (Quase idênticas) ---

# Raio da Terra em metros
EARTH_RADIUS_METERS = 6371000

def load_data(uploaded_file) -> pd.DataFrame:
    """
    Carrega os dados do log de telemetria de um arquivo enviado pelo Streamlit.
    """
    try:
        # O Streamlit 'uploaded_file' pode ser lido diretamente pelo pandas
        df = pd.read_csv(uploaded_file)
        
        # Validar colunas necessárias
        required_cols = ['lat', 'lng', 'vel']
        if not all(col in df.columns for col in required_cols):
            # 'st.error' mostra uma caixa de erro bonita no app
            st.error(f"O arquivo CSV deve conter as colunas: {required_cols}")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return pd.DataFrame()

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula a distância em metros entre dois pontos (Haversine)."""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = EARTH_RADIUS_METERS * c
    return distance

def process_telemetry_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o tempo decorrido e a aceleração (sem alterações)."""
    if df.empty:
        return pd.DataFrame()
    
    df_processed = df.copy()
    df_processed['vel_mps'] = df_processed['vel'] / 3.6
    df_processed['lat_prev'] = df_processed['lat'].shift(1)
    df_processed['lng_prev'] = df_processed['lng'].shift(1)
    df_processed['vel_mps_prev'] = df_processed['vel_mps'].shift(1)
    
    df_processed.loc[0, ['lat_prev', 'lng_prev', 'vel_mps_prev']] = \
        df_processed.loc[0, ['lat', 'lng', 'vel_mps']]
        
    df_processed['delta_distance_m'] = df_processed.apply(
        lambda row: calculate_haversine_distance(
            row['lat_prev'], row['lng_prev'], row['lat'], row['lng']
        ),
        axis=1
    )
    df_processed['avg_vel_mps'] = (df_processed['vel_mps'] + df_processed['vel_mps_prev']) / 2
    df_processed['delta_t_s'] = 0.0
    mask_vel_pos = df_processed['avg_vel_mps'] > 1e-6
    df_processed.loc[mask_vel_pos, 'delta_t_s'] = (
        df_processed.loc[mask_vel_pos, 'delta_distance_m'] /
        df_processed.loc[mask_vel_pos, 'avg_vel_mps']
    )
    df_processed['time_s'] = df_processed['delta_t_s'].cumsum()
    df_processed['delta_v_mps'] = df_processed['vel_mps'] - df_processed['vel_mps_prev']
    df_processed['acceleration_mpss'] = 0.0
    mask_time_pos = df_processed['delta_t_s'] > 1e-6
    df_processed.loc[mask_time_pos, 'acceleration_mpss'] = (
        df_processed.loc[mask_time_pos, 'delta_v_mps'] /
        df_processed.loc[mask_time_pos, 'delta_t_s']
    )
    df_processed.loc[0, 'acceleration_mpss'] = 0.0
    
    return df_processed

# --- Funções de Plotagem (MODIFICADAS) ---
# Em vez de salvar em arquivo, elas retornam o objeto 'figure'

def plot_velocity_vs_time(df: pd.DataFrame) -> plt.Figure:
    """Gera um gráfico de Velocidade vs. Tempo e retorna a figura."""
    if df.empty:
        return plt.Figure()

    fig, ax = plt.subplots(figsize=(12, 6)) # Cria uma figura e um eixo
    ax.plot(df['time_s'], df['vel'], marker='.', linestyle='-', markersize=4)
    ax.set_title('Velocidade vs. Tempo', fontsize=16)
    ax.set_xlabel('Tempo (s)', fontsize=12)
    ax.set_ylabel('Velocidade (km/h)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    return fig # Retorna a figura para o Streamlit

def plot_acceleration_vs_time(df: pd.DataFrame) -> plt.Figure:
    """Gera um gráfico de Aceleração vs. Tempo e retorna a figura."""
    if df.empty:
        return plt.Figure()

    fig, ax = plt.subplots(figsize=(12, 6)) # Cria uma figura e um eixo
    ax.plot(df['time_s'], df['acceleration_mpss'], marker='.', linestyle='-', markersize=4, color='orange')
    ax.set_title('Aceleração vs. Tempo', fontsize=16)
    ax.set_xlabel('Tempo (s)', fontsize=12)
    ax.set_ylabel('Aceleração (m/s²)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    return fig # Retorna a figura para o Streamlit


# --- Lógica Principal da Aplicação (O "main" do Streamlit) ---

# Título que aparece no topo da página web
st.title("Analisador de Log de Telemetria")

# Widget de Upload de Arquivo
# Isso cria um botão de "Browse files"
uploaded_file = st.file_uploader("Escolha um arquivo de log (.csv)", type="csv")

# O código abaixo só é executado DEPOIS que o usuário envia um arquivo
if uploaded_file is not None:
    
    df_raw = load_data(uploaded_file)
    
    if not df_raw.empty:
        # 'st.dataframe' mostra o seu DataFrame no app (ótimo para depurar)
        st.subheader("Dados Brutos Carregados (5 primeiras linhas)")
        st.dataframe(df_raw.head())

        # 'st.spinner' mostra uma mensagem de "carregando..."
        with st.spinner("Calculando e gerando gráficos... Por favor, aguarde."):
            
            df_processed = process_telemetry_data(df_raw)
            
            if not df_processed.empty:
                
                # Gerar os gráficos
                fig_vel = plot_velocity_vs_time(df_processed)
                fig_accel = plot_acceleration_vs_time(df_processed)

                # ... (vários códigos acima) ...

                # 'st.subheader' é como um subtítulo
                st.subheader("Gráfico de Velocidade vs. Tempo")
                # 'st.pyplot' é como o Streamlit exibe um gráfico do Matplotlib
                st.pyplot(fig_vel)

                st.subheader("Gráfico de Aceleração vs. Tempo")
                st.pyplot(fig_accel)

                # --- ADICIONE SEU NOVO CÓDIGO AQUI ---
                st.subheader("Mapa da Rota")
                st.map(df_processed, latitude='lat', longitude='lng')
                # ------------------------------------
                
                # 'st.success' mostra uma mensagem de sucesso
                st.success("Processamento concluído com sucesso!")
# ... (fim do código) ...
            
            else:
                st.error("Falha ao processar os dados.")
