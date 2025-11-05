import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt  # <-- 1. Importar o Altair

# --- Funções de Cálculo (Exatamente as mesmas) ---

# Raio da Terra em metros
EARTH_RADIUS_METERS = 6371000

def load_data(uploaded_file) -> pd.DataFrame:
    """Carrega os dados do log de telemetria."""
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['lat', 'lng', 'vel']
        if not all(col in df.columns for col in required_cols):
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

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula o rolamento (direção) em graus de um ponto para outro."""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2_rad - lon1_rad
    x = np.cos(lat2_rad) * np.sin(dLon)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dLon)
    bearing_rad = np.arctan2(x, y)
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360
    return bearing_deg

def process_telemetry_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o tempo decorrido e a aceleração."""
    if df.empty: return pd.DataFrame()
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
        ), axis=1
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
    df_processed['bearing'] = df_processed.apply(
        lambda row: calculate_bearing(
            row['lat_prev'], row['lng_prev'], row['lat'], row['lng']
        ), axis=1
    )
    df_processed['icon'] = 'arrow'
    return df_processed

# --- Lógica Principal da Aplicação ---

st.title("Analisador de Log de Telemetria")

uploaded_file = st.file_uploader("Escolha um arquivo de log (.csv)", type="csv")

if uploaded_file is not None:
    
    df_raw = load_data(uploaded_file)
    
    if not df_raw.empty:
        st.subheader("Dados Brutos Carregados (5 primeiras linhas)")
        st.dataframe(df_raw.head())

        with st.spinner("Calculando e gerando gráficos... Por favor, aguarde."):
            
            df_processed = process_telemetry_data(df_raw)
            
            if not df_processed.empty:
                
                # --- 2. MUDANÇA AQUI: de st.line_chart para st.altair_chart ---
                
                st.subheader("Gráfico de Velocidade vs. Tempo")
                
                # Criar o gráfico com Altair
                vel_chart = alt.Chart(df_processed).mark_line().encode(
                    # Configurar o eixo X: 'tickCount=10' pede "mais números"
                    x=alt.X('time_s', title='Tempo (s)', axis=alt.Axis(grid=True, tickCount=10)),
                    # Configurar o eixo Y
                    y=alt.Y('vel', title='Velocidade (km/h)', axis=alt.Axis(grid=True, tickCount=10)),
                    tooltip=['time_s', 'vel'] # Tooltip ao passar o rato
                ).interactive() # Permite zoom e arrastar
                
                # Exibir o gráfico
                st.altair_chart(vel_chart, use_container_width=True)


                st.subheader("Gráfico de Aceleração vs. Tempo")
                
                accel_chart = alt.Chart(df_processed).mark_line(color='orange').encode(
                    x=alt.X('time_s', title='Tempo (s)', axis=alt.Axis(grid=True, tickCount=10)),
                    y=alt.Y('acceleration_mpss', title='Aceleração (m/s²)', axis=alt.Axis(grid=True, tickCount=10)),
                    tooltip=['time_s', 'acceleration_mpss']
                ).interactive()
                
                st.altair_chart(accel_chart, use_container_width=True)
                
                # --- FIM DA MUDANÇA ---


                # O mapa Pydeck continua igual
                st.subheader("Mapa da Rota")
                
                ICON_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-arrow-up.png"
                icon_data = {"url": ICON_URL, "width": 128, "height": 128, "anchorY": 128}
                
                icon_layer = pdk.Layer(
                    "IconLayer", data=df_processed, get_icon="icon",
                    get_position="[lng, lat]", get_size=40, get_angle="bearing",
                    billboard=False, pickable=True, icon_atlas=icon_data["url"],
                    icon_mapping={"arrow": {"x": 0, "y": 0, "width": icon_data["width"], "height": icon_data["height"], "mask": True}}
                )
                
                path_data = df_processed[['lng', 'lat']].values.tolist()
                path_layer = pdk.Layer(
                    "PathLayer", data=[{"path": path_data, "name": "Rota"}],
                    get_path="path", get_width=2, width_min_pixels=2,
                    get_color=[255, 0, 0, 255],
                )

                mid_lat = df_processed['lat'].mean()
                mid_lng = df_processed['lng'].mean()
                
                view_state = pdk.ViewState(
                    latitude=mid_lat, longitude=mid_lng,
                    zoom=15, pitch=45,
                )

                st.pydeck_chart(pdk.Deck(
                    layers=[path_layer, icon_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/light-v9",
                    tooltip={"text": "Velocidade: {vel} km/h\nDireção: {bearing:.1f}°"}
                ))
                
                st.success("Processamento concluído com sucesso!")
            
            else:
                st.error("Falha ao processar os dados.")
