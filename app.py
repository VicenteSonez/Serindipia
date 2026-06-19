import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
from libpysal.weights import KNN
import esda
import json

# --- Page Config ---
st.set_page_config(
    page_title="Dashboard CASEN 2024",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for A4 printing ---
st.markdown("""
<style>
    /* Estilos para facilitar la impresión A4 */
    @media print {
        @page {
            size: A4 portrait;
            margin: 1cm;
        }
        .stApp {
            width: 21cm;
            background: white;
            color: black;
        }
        header, footer, .stDeployButton, .stToolbar {
            display: none !important;
        }
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0rem !important;
        padding-bottom: 0rem !important;
        color: #2c3e50;
    }
    h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.2rem !important;
        color: #2c3e50;
    }
    .stMarkdown p {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Análisis multidimensional de ingresos en Chile • CASEN 2024")
st.markdown("**Autores**: Vicente Soñez Ibañez, Rafael Fernández Castellón, Pedro Dañobeytia Gómez")

# --- Data Loading ---
@st.cache_data
def load_data():
    import os
    current_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.dirname(current_dir)
    
    # Búsqueda resiliente (útil para servidores Linux que son sensibles a mayúsculas/minúsculas)
    posibles_rutas_parquet = [
        os.path.join(base_dir, 'Data', 'casen_2024.parquet'),
        os.path.join(base_dir, 'data', 'casen_2024.parquet'),
        os.path.join(current_dir, 'Data', 'casen_2024.parquet'),
        os.path.join(current_dir, 'data', 'casen_2024.parquet')
    ]
    
    parquet_path = None
    for ruta in posibles_rutas_parquet:
        if os.path.exists(ruta):
            parquet_path = ruta
            break
            
    if parquet_path is None:
        raise FileNotFoundError("No se encontró 'casen_2024.parquet'. Por favor asegúrate de haber subido la carpeta 'Data' a tu repositorio de GitHub.")
        
    geojson_path = parquet_path.replace('casen_2024.parquet', 'regiones.geojson')

    df = pd.read_parquet(parquet_path)
    try:
        gdf = gpd.read_file(geojson_path, engine='pyogrio')
    except:
        gdf = gpd.read_file(geojson_path)
        
    # Filtrar Isla de Pascua y Juan Fernández para visualización de Chile Continental
    from shapely.geometry import MultiPolygon
    new_geoms = []
    for geom in gdf.geometry:
        if geom is None:
            new_geoms.append(None)
            continue
        if geom.geom_type == 'MultiPolygon':
            clean_polys = [p for p in geom.geoms if p.bounds[0] >= -76]
            new_geoms.append(MultiPolygon(clean_polys) if clean_polys else None)
        elif geom.geom_type == 'Polygon':
            new_geoms.append(geom if geom.bounds[0] >= -76 else None)
        else:
            new_geoms.append(geom)
    gdf = gdf.copy()
    gdf.geometry = new_geoms
    gdf = gdf[gdf.geometry.notnull()]
    
    return df, gdf

with st.spinner("Cargando datos..."):
    df_full, gdf_regiones = load_data()


# ==========================================
# ROW 1
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Flujo educativo a quintiles")
    df_alluvial = df_full[['educc', 'qaut', 'activ']].copy()
    df_alluvial = df_alluvial[(df_alluvial['educc'] != -88) & (df_alluvial['activ'] == 1)].copy()

    educc_map = {
        0: 'Educ. Media Inc. o inferior', 1: 'Educ. Media Inc. o inferior',
        2: 'Educ. Media Inc. o inferior', 3: 'Educ. Media Inc. o inferior',
        4: 'Educ. Media Comp. / Sup. Inc.', 5: 'Educ. Media Comp. / Sup. Inc.',
        6: 'Educ. Superior Completa'
    }
    qaut_map = {1: 'Quintil 1', 2: 'Quintil 2', 3: 'Quintil 3', 4: 'Quintil 4', 5: 'Quintil 5'}

    df_alluvial['educc_label'] = df_alluvial['educc'].map(educc_map)
    df_alluvial['qaut_label'] = df_alluvial['qaut'].map(qaut_map)
    df_alluvial = df_alluvial.dropna(subset=['educc_label', 'qaut_label'])

    flujos = df_alluvial.groupby(['educc_label', 'qaut_label']).size().reset_index(name='cantidad')

    color_flujos_dict = {
        'Educ. Media Inc. o inferior': 'rgba(231, 76, 60, 0.45)',      # Rojo
        'Educ. Media Comp. / Sup. Inc.': 'rgba(52, 152, 219, 0.45)',  # Azul
        'Educ. Superior Completa': 'rgba(46, 204, 113, 0.45)'          # Verde
    }
    color_nodos_dict = {
        'Quintil 1': 'rgba(149, 165, 166, 0.8)', 'Quintil 2': 'rgba(149, 165, 166, 0.8)',
        'Quintil 3': 'rgba(149, 165, 166, 0.8)', 'Quintil 4': 'rgba(149, 165, 166, 0.8)',
        'Quintil 5': 'rgba(149, 165, 166, 0.8)',
        'Educ. Media Inc. o inferior': 'rgba(231, 76, 60, 0.8)',
        'Educ. Media Comp. / Sup. Inc.': 'rgba(52, 152, 219, 0.8)',
        'Educ. Superior Completa': 'rgba(46, 204, 113, 0.8)'
    }

    colores_enlaces = flujos['educc_label'].map(color_flujos_dict).tolist()
    nodos_origen = ['Educ. Media Inc. o inferior', 'Educ. Media Comp. / Sup. Inc.', 'Educ. Superior Completa']
    nodos_destino = [qaut_map[i] for i in range(1, 6)]
    todos_los_nodos = nodos_origen + nodos_destino

    indice_nodos = {nodo: i for i, nodo in enumerate(todos_los_nodos)}
    colores_nodos_lista = [color_nodos_dict[nodo] for nodo in todos_los_nodos]

    fig1 = go.Figure(data=[go.Sankey(
        textfont=dict(size=11, family="Arial, sans-serif", color="black"),
        node = dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=todos_los_nodos, color=colores_nodos_lista),
        link = dict(
            source=flujos['educc_label'].map(indice_nodos).tolist(),
            target=flujos['qaut_label'].map(indice_nodos).tolist(),
            value=flujos['cantidad'].tolist(),
            color=colores_enlaces,
            hovertemplate='%{source.label} → %{target.label}<br>Cantidad: %{value:,.0f}<extra></extra>'
        )
    )])
    fig1.update_layout(height=350, margin=dict(t=10, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("2. Sueldo por área y género")
    df2 = df_full[['cinef13_area', 'sexo', 'ytrabajocor']].copy().dropna()
    df2 = df2[(df2['cinef13_area'] >= 0) & (df2['sexo'].isin([1, 2]))]

    cine_map = {
        1.0: 'Salud y Bienestar', 2.0: 'Ingeniería y Const.', 3.0: 'Educación', 4.0: 'Servicios',
        5.0: 'Admin. y Derecho', 6.0: 'Ciencias Sociales', 7.0: 'Ciencias Naturales',
        8.0: 'Agric. y Veterinaria', 9.0: 'Informática (TIC)', 10.0: 'Artes y Humanidades', 11.0: 'Cs. Básicas'
    }
    df2['area_label'] = df2['cinef13_area'].map(cine_map)
    df2['genero_label'] = df2['sexo'].map({1: 'Hombre', 2: 'Mujer'})
    df2 = df2.dropna(subset=['area_label'])

    stats = df2.groupby(['area_label', 'genero_label'], as_index=False)['ytrabajocor'].agg(promedio='mean', tamano='count')
    area_order = df2.groupby('area_label')['ytrabajocor'].mean().sort_values(ascending=True).index
    stats['area_label'] = pd.Categorical(stats['area_label'], categories=area_order, ordered=True)
    stats = stats.sort_values('area_label')

    fig2 = go.Figure()

    for area in area_order:
        area_data = stats[stats['area_label'] == area]
        if len(area_data) == 2:
            val_h = area_data[area_data['genero_label'] == 'Hombre']['promedio'].values[0]
            val_m = area_data[area_data['genero_label'] == 'Mujer']['promedio'].values[0]
            fig2.add_trace(go.Scatter(
                x=[val_m, val_h], y=[area, area],
                mode='lines',
                line=dict(color='#C8B19F', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

    h_data = stats[stats['genero_label'] == 'Hombre']
    fig2.add_trace(go.Scatter(
        x=h_data['promedio'], y=h_data['area_label'],
        mode='markers',
        marker=dict(color='#2B5B84', size=12, line=dict(color='white', width=1)),
        name='Hombre',
        text=h_data['tamano'],
        hovertemplate="<b>Área:</b> %{y}<br><b>Ingreso:</b> $%{x:,.0f}<br><b>Muestra:</b> %{text}<extra></extra>"
    ))

    m_data = stats[stats['genero_label'] == 'Mujer']
    fig2.add_trace(go.Scatter(
        x=m_data['promedio'], y=m_data['area_label'],
        mode='markers',
        marker=dict(color='#C23B22', size=12, line=dict(color='white', width=1)),
        name='Mujer',
        text=m_data['tamano'],
        hovertemplate="<b>Área:</b> %{y}<br><b>Ingreso:</b> $%{x:,.0f}<br><b>Muestra:</b> %{text}<extra></extra>"
    ))
    
    fig2.update_layout(
        xaxis=dict(tickformat='$,.0f', title='Sueldo Promedio (CLP)', showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1, griddash='dot'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1, griddash='dot', tickfont=dict(size=11)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=350, margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig2, use_container_width=True)


# ==========================================
# ROW 2
# ==========================================
col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Brecha por macrozona y quintil")
    df3 = df_full.dropna(subset=['region', 'sexo', 'yoprcor']).copy()
    macrozona_mapping = {
        1: 'Z. Norte', 2: 'Z. Norte', 3: 'Z. Centro Norte', 4: 'Z. Centro Norte',
        5: 'Z. Centro', 6: 'Z. Centro', 7: 'Z. Centro Sur', 8: 'Z. Centro Sur',
        9: 'Z. Sur', 10: 'Z. Sur', 11: 'Z. Sur', 12: 'Z. Sur',
        13: 'Z. Centro', 14: 'Z. Sur', 15: 'Z. Norte', 16: 'Z. Centro Sur'
    }
    df3['Macrozona'] = df3['region'].map(macrozona_mapping)
    macrozona_order = ['Z. Norte', 'Z. Centro Norte', 'Z. Centro', 'Z. Centro Sur', 'Z. Sur']
    df3['Macrozona'] = pd.Categorical(df3['Macrozona'], categories=macrozona_order, ordered=True)
    df3['Genero'] = df3['sexo'].map({1: 'Hombre', 2: 'Mujer'})
    map_qaut = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5'}
    df3['qaut_label'] = df3['qaut'].map(map_qaut)
    df3 = df3.dropna(subset=['qaut_label'])

    pivot = df3.pivot_table(values='yoprcor', index='Macrozona', columns=['qaut_label', 'Genero'], aggfunc='mean', observed=False)
    gap_df = pd.DataFrame(index=macrozona_order)
    for cat in pivot.columns.levels[0]:
        try:
            gap_df[cat] = ((pivot[cat]['Hombre'] - pivot[cat]['Mujer']) / pivot[cat]['Hombre']) * 100
        except: pass
    gap_df = gap_df.dropna(how='all', axis=1).T

    fig3 = px.imshow(
        gap_df,
        text_auto=".1f",
        color_continuous_scale="YlOrRd",
        zmin=0, zmax=35,
        labels=dict(color="Brecha (%)")
    )
    fig3.update_layout(
        xaxis_title="Macro-Zona",
        yaxis_title="Quintil",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(t=20, b=20, l=10, r=10),
        coloraxis_colorbar=dict(title="Brecha (%)")
    )
    st.plotly_chart(fig3, use_container_width=True)


with col4:
    st.subheader("4. Educación por zona")
    educc_map_2 = {
        0: 'Sin Educ. Formal', 1: 'Básica Incom.', 2: 'Básica Comp.',
        3: 'Media Incom.', 4: 'Media Comp.', 5: 'Sup. Incom.', 6: 'Sup. Comp.'
    }
    zona_map = {1: 'Urbano', 2: 'Rural'}

    df_educ = df_full[(df_full['educc'] >= 0) & (df_full['area'].isin([1, 2]))].copy()
    df_educ['Nivel Educacional'] = df_educ['educc'].map(educc_map_2)
    df_educ['Zona'] = df_educ['area'].map(zona_map)

    educ_zona_counts = df_educ.groupby(['Zona', 'Nivel Educacional']).size().reset_index(name='Cantidad')

    fig4 = px.treemap(
        educ_zona_counts,
        path=['Zona', 'Nivel Educacional'],
        values='Cantidad',
        color='Cantidad',
        color_continuous_scale='Oranges'
    )
    fig4.update_layout(height=350, margin=dict(t=10, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    fig4.update_traces(hovertemplate='<b>%{label}</b><br>Cantidad: %{value:,.0f}<extra></extra>')
    st.plotly_chart(fig4, use_container_width=True)


# ==========================================
# ROW 3
# ==========================================
col5, col6 = st.columns(2)

with col5:
    st.subheader("5. Crecimiento de brecha por edad")
    df_ocupados = df_full[(df_full['activ'] == 1.0) & (df_full['yoprcor'] > 0) & (df_full['edad'] >= 18) & (df_full['edad'] <= 75)].copy()

    df_brecha = df_ocupados.groupby(['edad', 'sexo'], observed=True)['yoprcor'].mean().unstack()
    df_brecha.columns = ['Hombre', 'Mujer']
    df_brecha = df_brecha.reset_index()

    df_brecha['Hombre_suave'] = df_brecha['Hombre'].rolling(window=3, center=True, min_periods=1).mean()
    df_brecha['Mujer_suave'] = df_brecha['Mujer'].rolling(window=3, center=True, min_periods=1).mean()

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=df_brecha['edad'], y=df_brecha['Mujer_suave'], name='Mujeres',
        mode='lines', line=dict(color='#C23B22', width=3, shape='spline', smoothing=1.3),
        hovertemplate='<b>Edad:</b> %{x} años<br><b>Ingreso:</b> $%{y:,.0f}<extra></extra>'
    ))
    fig5.add_trace(go.Scatter(
        x=df_brecha['edad'], y=df_brecha['Hombre_suave'], name='Hombres',
        mode='lines', line=dict(color='#2B5B84', width=3, shape='spline', smoothing=1.3),
        fill='tonexty', fillcolor='rgba(43, 91, 132, 0.15)',
        hovertemplate='<b>Edad:</b> %{x} años<br><b>Ingreso:</b> $%{y:,.0f}<extra></extra>'
    ))
    fig5.update_layout(
        xaxis=dict(title='Edad (Años)', showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1, griddash='dot', dtick=10),
        yaxis=dict(title='Ingreso Promedio (CLP)', tickformat='$,.0f', showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1, griddash='dot'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        height=400, margin=dict(t=20, l=10, r=10, b=10)
    )
    st.plotly_chart(fig5, use_container_width=True)


with col6:
    st.subheader("6. Autocorrelacion espacial por ingreso")
    
    # Preparar datos
    df_r = df_full.groupby('region')['yoprcor'].mean().reset_index()
    gdf = gdf_regiones.merge(df_r, left_on='codregion', right_on='region', how='inner')
    
    # Calcular Moran's I
    w = KNN.from_dataframe(gdf, k=2)
    w.transform = 'r'
    y = gdf['yoprcor'].values
    moran = esda.Moran(y, w)
    
    # Extraer geojson para Plotly Mapbox
    geojson_data = json.loads(gdf.to_json())
    
    # Graficar mapa en Plotly
    fig6 = px.choropleth_mapbox(
        gdf,
        geojson=geojson_data,
        locations='codregion',
        featureidkey="properties.codregion",
        color='yoprcor',
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=2.8,
        center={"lat": -37.5, "lon": -72.0}, # Centro de Chile aprox
        opacity=0.75,
        hover_name='Region',
        hover_data={'codregion': False, 'yoprcor': ':,.0f'}
    )
    
    fig6.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title=dict(text=f"<b>Índice de Moran: {moran.I:.3f} (p-val: {moran.p_sim:.3f})</b>", font=dict(size=14, color='#2c3e50')),
        coloraxis_colorbar=dict(title="Ingreso Promedio", tickformat="$,.0f", thickness=15),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig6.update_traces(hovertemplate='<b>%{hovertext}</b><br>Ingreso: $%{z:,.0f}<extra></extra>')
    st.plotly_chart(fig6, use_container_width=True)


