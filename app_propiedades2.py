import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import joblib
from datetime import datetime

# Cargar datos reales desde el CSV
df = pd.read_csv('datos_entrenamiento_completo.csv')

# Función para obtener coordenadas reales del dataset
def get_city_coordinates_from_dataset(state, city):
    """Obtener coordenadas REALES del dataset para la ciudad específica"""
    try:
        # Filtrar el dataset por estado y ciudad
        city_data = df[(df['state'] == state) & (df['cityname'] == city)]
        
        if not city_data.empty:
            # Tomar el primer registro que tenga coordenadas válidas
            valid_coords = city_data[(city_data['latitude'] != 0.0) & 
                                   (city_data['longitude'] != 0.0)]
            
            if not valid_coords.empty:
                lat = valid_coords['latitude'].iloc[0]
                lon = valid_coords['longitude'].iloc[0]
                print(f"COORDENADAS REALES ENCONTRADAS: {city}, {state} - Lat: {lat}, Lon: {lon}")
                return {'latitude': lat, 'longitude': lon}
        
        # Si no encontramos coordenadas específicas, usar las del estado
        print(f"No se encontraron coordenadas específicas para {city}, usando coordenadas del estado")
        return get_state_coordinates(state)
        
    except Exception as e:
        print(f"Error obteniendo coordenadas: {e}")
        return get_state_coordinates(state)

def get_state_coordinates(state):
    """Coordenadas por estado como fallback"""
    state_coordinates = {
        'AL': {'latitude': 32.806671, 'longitude': -86.791130},
        'AK': {'latitude': 61.370716, 'longitude': -152.404419},
        'AZ': {'latitude': 33.729759, 'longitude': -111.431221},
        'AR': {'latitude': 34.969704, 'longitude': -92.373123},
        'CA': {'latitude': 36.116203, 'longitude': -119.681564},
        'CO': {'latitude': 39.059811, 'longitude': -105.311104},
        'CT': {'latitude': 41.597782, 'longitude': -72.755371},
        'DE': {'latitude': 39.318523, 'longitude': -75.507141},
        'FL': {'latitude': 27.766279, 'longitude': -81.686783},
        'GA': {'latitude': 33.040619, 'longitude': -83.643074},
        'HI': {'latitude': 21.094318, 'longitude': -157.498337},
        'ID': {'latitude': 44.240459, 'longitude': -114.478828},
        'IL': {'latitude': 40.349457, 'longitude': -88.986137},
        'IN': {'latitude': 39.849426, 'longitude': -86.258278},
        'IA': {'latitude': 42.011539, 'longitude': -93.210526},
        'KS': {'latitude': 38.526600, 'longitude': -96.726486},
        'KY': {'latitude': 37.668140, 'longitude': -84.670067},
        'LA': {'latitude': 31.169546, 'longitude': -91.867805},
        'ME': {'latitude': 44.693947, 'longitude': -69.381927},
        'MD': {'latitude': 39.063946, 'longitude': -76.802101},
        'MA': {'latitude': 42.230171, 'longitude': -71.530106},
        'MI': {'latitude': 43.326618, 'longitude': -84.536095},
        'MN': {'latitude': 45.694454, 'longitude': -93.900192},
        'MS': {'latitude': 32.741646, 'longitude': -89.678696},
        'MO': {'latitude': 38.456085, 'longitude': -92.288368},
        'MT': {'latitude': 46.921925, 'longitude': -110.454353},
        'NE': {'latitude': 41.125370, 'longitude': -98.268082},
        'NV': {'latitude': 38.313515, 'longitude': -117.055374},
        'NH': {'latitude': 43.452492, 'longitude': -71.563896},
        'NJ': {'latitude': 40.298904, 'longitude': -74.521011},
        'NM': {'latitude': 34.840515, 'longitude': -106.248482},
        'NY': {'latitude': 42.165726, 'longitude': -74.948051},
        'NC': {'latitude': 35.630066, 'longitude': -79.806419},
        'ND': {'latitude': 47.528912, 'longitude': -99.784012},
        'OH': {'latitude': 40.388783, 'longitude': -82.764915},
        'OK': {'latitude': 35.565342, 'longitude': -96.928917},
        'OR': {'latitude': 44.572021, 'longitude': -122.070938},
        'PA': {'latitude': 40.590752, 'longitude': -77.209755},
        'RI': {'latitude': 41.680893, 'longitude': -71.511780},
        'SC': {'latitude': 33.856892, 'longitude': -80.945007},
        'SD': {'latitude': 44.299782, 'longitude': -99.438828},
        'TN': {'latitude': 35.747845, 'longitude': -86.692345},
        'TX': {'latitude': 31.054487, 'longitude': -97.563461},
        'UT': {'latitude': 40.150032, 'longitude': -111.862434},
        'VT': {'latitude': 44.045876, 'longitude': -72.710686},
        'VA': {'latitude': 37.769337, 'longitude': -78.169968},
        'WA': {'latitude': 47.400902, 'longitude': -121.490494},
        'WV': {'latitude': 38.491226, 'longitude': -80.954453},
        'WI': {'latitude': 44.268543, 'longitude': -89.616508},
        'WY': {'latitude': 42.755966, 'longitude': -107.302490}
    }
    
    if state in state_coordinates:
        return state_coordinates[state]
    else:
        return {'latitude': 39.8283, 'longitude': -98.5795}

class ProbabilityModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.available_features = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = joblib.load('modelo_entrenado.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            self.available_features = joblib.load('available_features.pkl')
            print("Modelo de probabilidad cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
    
    def predict_probability(self, input_data):
        if self.model is None:
            return {"error": "Modelo no disponible"}
        
        try:
            # Obtener coordenadas basadas en la ciudad seleccionada
            city_coords = get_city_coordinates_from_dataset(input_data['state'], input_data['city'])
            
            input_dict = {
                'bathrooms': int(input_data['bathrooms']),
                'bedrooms': int(input_data['bedrooms']),
                'square_feet': int(input_data['square_feet']),
                'latitude': city_coords['latitude'],
                'longitude': city_coords['longitude'],
                'has_Gym': 1 if input_data.get('has_gym') else 0,
                'has_Parking': 1 if input_data.get('has_parking') else 0,
                'has_Pool': 1 if input_data.get('has_pool') else 0,
                'has_Internet_Access': 1 if input_data.get('has_internet') else 0,
                'has_Elevator': 1 if input_data.get('has_elevator') else 0,
                'allows_Dogs': 1 if input_data.get('allows_dogs') else 0,
                'allows_Cats': 1 if input_data.get('allows_cats') else 0,
                'match': 1
            }
            
            # Añadir características codificadas
            categorical_cols = ['category', 'price_type', 'cityname', 'state', 'zone']
            for col in categorical_cols:
                encoded_col = col + '_encoded'
                if encoded_col in self.available_features:
                    input_dict[encoded_col] = 0
            
            # Crear DataFrame
            input_df = pd.DataFrame([input_dict], columns=self.available_features)
            
            # Realizar predicción de probabilidad
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(input_df)[0][1]
            else:
                if hasattr(self.model, 'coef_'):
                    input_scaled = self.scaler.transform(input_df)
                    raw_prediction = self.model.predict(input_scaled)[0]
                else:
                    raw_prediction = self.model.predict(input_df)[0]
                probability = self.normalize_to_probability(raw_prediction)
            
            return {
                "probability": probability * 100,
                "latitude": city_coords['latitude'],
                "longitude": city_coords['longitude'],
                "city": input_data['city']
            }
            
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            return {
                "probability": 25,
                "latitude": 39.8283,
                "longitude": -98.5795,
                "city": "Unknown"
            }
    
    def normalize_to_probability(self, value):
        normalized = max(0, min(1, (value - 500) / 2000))
        return normalized

probability_model = ProbabilityModel()

# Extraer estados y ciudades únicos del dataset real
estados_unicos = sorted(df['state'].dropna().unique())
ciudades_por_estado = {}

for estado in estados_unicos:
    ciudades_estado = sorted(df[df['state'] == estado]['cityname'].dropna().unique())
    ciudades_por_estado[estado] = ciudades_estado

# Paleta de colores empresarial
DARK_BG = '#171A21'
LIGHT_BG = '#2D2F36'
ACCENT = '#FFA400'
BASE_MAP = '#7A93AC'
CARD_BG = '#23262F'
TEXT_GRAY = '#B8BCC8'
BORDER_COLOR = '#3A3D45'

# Crear app
app = dash.Dash(__name__)
server = app.server

# Función para crear el mapa inicial sin distorsión
def create_initial_map_figure(selected_state=None):
    estados_reales = list(ciudades_por_estado.keys())
    
    z_values = [1 if estado == selected_state else 0.3 for estado in estados_reales]
    
    colorscale = [[0, BASE_MAP], [1, ACCENT]]
    
    fig = go.Figure(data=go.Choropleth(
        locations=estados_reales,
        z=z_values,
        locationmode='USA-states',
        colorscale=colorscale,
        showscale=False,
        marker_line_color='white',
        marker_line_width=1,
        hovertemplate='<b>%{location}</b><extra></extra>'
    ))
    
    fig.update_geos(
        scope='usa',
        projection_type='albers usa',
        showlakes=True,
        lakecolor='blue',
        showrivers=True,
        rivercolor='blue',
        bgcolor=CARD_BG,
        landcolor='lightgray',
        subunitcolor='white'
    )
    
    fig.update_layout(
        geo=dict(
            bgcolor=CARD_BG,
            landcolor='lightgray',
            subunitcolor='white'
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        uirevision='constant'
    )
    
    return fig

def create_city_map(lat, lon, city_name, probability, state):
    """Función COMPLETAMENTE NUEVA - Mapa de ciudad que FUNCIONA con coordenadas reales"""
    
    print(f"Creando mapa REAL para: {city_name}, {state} - Lat: {lat}, Lon: {lon}")
    
    # Crear figura desde cero
    fig = go.Figure()
    
    # Añadir marcador en las coordenadas REALES
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(
            size=25,
            color=ACCENT,
            opacity=0.9
        ),
        text=[f"{city_name}, {state}"],
        textposition="top center",
        textfont=dict(
            size=14,
            color='white',
            family="Arial Black"
        ),
        hovertemplate=(
            f"<b>{city_name}, {state}</b><br>"
            f"Probability: {probability:.1f}%<br>"
            f"Coordinates: {lat:.4f}°N, {lon:.4f}°W<br>"
            "<extra></extra>"
        )
    ))
    
    # Configuración CRÍTICA - Centrar en las coordenadas REALES
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),  
            zoom=10,  # Zoom más cercano para ver la ciudad
            bearing=0,
            pitch=0
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        showlegend=False
    )
    
    return fig

# Calcular rangos de precios reales del dataset
precio_min = int(df['price'].min())
precio_max = int(df['price'].max())

slider_min = 0
slider_max = 5000
slider_step = 100

# Estilos para dropdowns
input_style = {
    'width': '100%',
    'padding': '12px',
    'backgroundColor': CARD_BG,
    'border': f'1px solid {BORDER_COLOR}',
    'borderRadius': '8px',
    'color': 'white',
    'fontSize': '14px',
    'marginBottom': '15px',
    'transition': 'all 0.3s ease'
}

dropdown_style = {
    'marginBottom': '15px',
    'color': 'white'
}

label_style = {
    'color': TEXT_GRAY,
    'fontSize': '13px',
    'fontWeight': '600',
    'marginBottom': '8px',
    'display': 'block',
    'textTransform': 'uppercase',
    'letterSpacing': '0.5px'
}

section_title_style = {
    'color': ACCENT,
    'fontSize': '15px',
    'fontWeight': 'bold',
    'marginTop': '25px',
    'marginBottom': '15px',
    'borderBottom': f'2px solid {ACCENT}',
    'paddingBottom': '8px',
    'display': 'inline-block',
    'width': '100%'
}

# Layout principal
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1([
                'Predictive Model of Probability of Existence of Properties'
            ], style={
                'textAlign': 'center',
                'color': 'white',
                'margin': 0,
                'fontWeight': '700',
                'fontSize': '32px',
                'letterSpacing': '1px'
            }),
            html.P('Intelligent Prediction of Property Existence Based on User Preferences',
                   style={
                       'textAlign': 'center',
                       'color': TEXT_GRAY,
                       'margin': '10px 0 0 0',
                       'fontSize': '16px',
                       'fontWeight': '400'
                   })
        ])
    ], style={
        'background': f'linear-gradient(135deg, {DARK_BG} 0%, #1f2229 100%)',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
        'padding': '30px 20px'
    }),
    
    # Contenido principal
    html.Div([
        html.Div([
            # Columna Izquierda
            html.Div([
                # Mapa de Estados
                html.Div([
                    html.Div([
                        html.H3('States Map', style={'color': 'white', 'fontSize': '18px', 'fontWeight': '700', 'margin': '0 0 5px 0'}),
                        html.P(style={'color': TEXT_GRAY, 'fontSize': '13px', 'margin': 0})
                    ]),
                    dcc.Graph(
                        id='mapa_estados',
                        figure=create_initial_map_figure(),
                        config={'displayModeBar': False},
                        style={'borderRadius': '12px', 'overflow': 'hidden', 'width': '100%', 'height': '100%'}
                    )
                ], style={
                    'backgroundColor': LIGHT_BG,
                    'padding': '25px',
                    'borderRadius': '16px',
                    'marginBottom': '25px',
                    'boxShadow': '0 8px 24px rgba(0,0,0,0.2)',
                    'border': f'1px solid {BORDER_COLOR}',
                    'height': '400px'
                }),
                
                # Mapa de Ciudad
                html.Div([
                    html.Div([
                        html.H3('City Location Map', style={'color': 'white', 'fontSize': '18px', 'fontWeight': '700', 'margin': '0 0 5px 0'}),
                        html.P('Selected city coordinates and location - Zoom and pan enabled', style={'color': TEXT_GRAY, 'fontSize': '13px', 'margin': 0})
                    ]),
                    dcc.Graph(
                        id='mapa_ciudad',
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'doubleClick': 'reset+autosize',
                            'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'resetScale2d'],
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            'displaylogo': False,
                            'responsive': True
                        },
                        style={
                            'borderRadius': '12px', 
                            'overflow': 'hidden',
                            'width': '100%',
                            'height': '100%',
                            'minHeight': '300px'
                        }
                    )
                ], style={
                    'backgroundColor': LIGHT_BG,
                    'padding': '25px',
                    'borderRadius': '16px',
                    'marginBottom': '25px',
                    'boxShadow': '0 8px 24px rgba(0,0,0,0.2)',
                    'border': f'1px solid {BORDER_COLOR}',
                    'display': 'none',
                    'height': '350px'
                }, id='mapa_ciudad_container'),
                
                # Range Slider de Precios
                html.Div([
                    html.Div([
                        html.H3('Price Range', style={'color': 'white', 'fontSize': '18px', 'fontWeight': '700', 'margin': '0 0 5px 0'}),
                        html.P(f'Real data range: ${precio_min} - ${precio_max}', style={'color': TEXT_GRAY, 'fontSize': '13px', 'margin': 0})
                    ]),
                    dcc.RangeSlider(
                        id='range_precio',
                        min=slider_min,
                        max=slider_max,
                        step=slider_step,
                        value=[1500, 2500],
                        marks={
                            0: {'label': '$0', 'style': {'color': TEXT_GRAY, 'fontSize': '12px', 'fontWeight': '600'}},
                            1000: {'label': '$1000', 'style': {'color': TEXT_GRAY, 'fontSize': '12px', 'fontWeight': '600'}},
                            2000: {'label': '$2000', 'style': {'color': TEXT_GRAY, 'fontSize': '12px', 'fontWeight': '600'}},
                            3000: {'label': '$3000', 'style': {'color': TEXT_GRAY, 'fontSize': '12px', 'fontWeight': '600'}},
                            4000: {'label': '$4000', 'style': {'color': TEXT_GRAY, 'fontSize': '12px', 'fontWeight': '600'}},
                            5000: {'label': '$5000', 'style': {'color': TEXT_GRAY, 'fontSize': '12px', 'fontWeight': '600'}}
                        },
                        tooltip={"placement": "bottom", "always_visible": True},
                        className='custom-range-slider'
                    ),
                    html.Div(id='precio_display', style={
                        'textAlign': 'center',
                        'color': ACCENT,
                        'marginTop': '30px',
                        'fontSize': '24px',
                        'fontWeight': '700',
                        'padding': '15px',
                        'backgroundColor': CARD_BG,
                        'borderRadius': '12px',
                        'border': f'2px solid {ACCENT}'
                    })
                ], style={
                    'backgroundColor': LIGHT_BG,
                    'padding': '25px',
                    'borderRadius': '16px',
                    'boxShadow': '0 8px 24px rgba(0,0,0,0.2)',
                    'border': f'1px solid {BORDER_COLOR}'
                })
            ], className='column-left'),
            
            # Columna Derecha
            html.Div([
                html.Div([
                    html.Div([
                        html.H3('Search Settings', style={'color': 'white', 'fontSize': '18px', 'fontWeight': '700', 'margin': '0 0 5px 0'}),
                        html.P('Complete the details of the desired property', style={'color': TEXT_GRAY, 'fontSize': '13px', 'margin': 0})
                    ]),
                    
                    # Dropdowns
                    html.Div([
                        html.Label('State', style=label_style),
                        dcc.Dropdown(
                            id='input_estado',
                            options=[{'label': estado, 'value': estado} for estado in ciudades_por_estado.keys()],
                            value=list(ciudades_por_estado.keys())[0] if ciudades_por_estado else None,
                            style=dropdown_style,
                            placeholder='Select a state',
                            className='custom-dropdown'
                        ),
                    ]),
                    
                    html.Div([
                        html.Label('City', style=label_style),
                        dcc.Dropdown(
                            id='input_ciudad',
                            style=dropdown_style,
                            placeholder='Select a city',
                            className='custom-dropdown'
                        ),
                    ]),
                    
                    html.Div(style=section_title_style, children='Amenities'),
                    dcc.Checklist(
                        id='input_amenities',
                        options=[
                            {'label': 'Gym', 'value': 'gym'},
                            {'label': 'Parking', 'value': 'parking'},
                            {'label': 'Pool', 'value': 'pool'},
                            {'label': 'Internet Access', 'value': 'internet'},
                            {'label': 'Elevator', 'value': 'elevator'}
                        ],
                        value=[],
                        style={'color': 'white', 'marginBottom': '20px'},
                        labelStyle={
                            'display': 'block',
                            'marginBottom': '10px',
                            'fontSize': '14px',
                            'cursor': 'pointer',
                            'padding': '8px 12px',
                            'backgroundColor': CARD_BG,
                            'borderRadius': '8px',
                            'border': f'1px solid {BORDER_COLOR}',
                            'transition': 'all 0.3s ease'
                        },
                        inputStyle={'marginRight': '10px'}
                    ),
                    
                    html.Div(style=section_title_style, children='Pets Allowed'),
                    dcc.Checklist(
                        id='input_pets',
                        options=[
                            {'label': 'Dogs', 'value': 'dogs'},
                            {'label': 'Cats', 'value': 'cats'}
                        ],
                        value=[],
                        style={'color': 'white', 'marginBottom': '20px'},
                        labelStyle={
                            'display': 'block',
                            'marginBottom': '10px',
                            'fontSize': '14px',
                            'cursor': 'pointer',
                            'padding': '8px 12px',
                            'backgroundColor': CARD_BG,
                            'borderRadius': '8px',
                            'border': f'1px solid {BORDER_COLOR}',
                            'transition': 'all 0.3s ease'
                        },
                        inputStyle={'marginRight': '10px'}
                    ),
                    
                    html.Div(style=section_title_style, children='Specifications'),
                    
                    html.Div([
                        html.Div([
                            html.Label('Area (sq ft)', style=label_style),
                            dcc.Input(
                                id='input_area',
                                type='number',
                                min=int(df['square_feet'].min()),
                                max=int(df['square_feet'].max()),
                                value=800,
                                style=input_style
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label('Bedrooms', style=label_style),
                            dcc.Input(
                                id='input_habitaciones',
                                type='number',
                                min=int(df['bedrooms'].min()),
                                max=int(df['bedrooms'].max()),
                                value=2,
                                style=input_style
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label('Bathrooms', style=label_style),
                            dcc.Input(
                                id='input_banos',
                                type='number',
                                min=int(df['bathrooms'].min()),
                                max=int(df['bathrooms'].max()),
                                value=1.5,
                                step=0.5,
                                style=input_style
                            ),
                        ], style={'width': '32%', 'display': 'inline-block'})
                    ], style={'marginBottom': '25px'}),
                    
                    html.Button('CALCULATE PROBABILITY',
                        id='boton_calcular',
                        n_clicks=0,
                        style={
                            'backgroundColor': ACCENT,
                            'color': 'white',
                            'fontWeight': '700',
                            'width': '100%',
                            'padding': '18px',
                            'fontSize': '16px',
                            'border': 'none',
                            'borderRadius': '12px',
                            'cursor': 'pointer',
                            'boxShadow': '0 4px 15px rgba(255, 164, 0, 0.3)',
                            'transition': 'all 0.3s ease',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px'
                        }
                    )
                ], style={
                    'backgroundColor': LIGHT_BG,
                    'padding': '25px',
                    'borderRadius': '16px',
                    'marginBottom': '25px',
                    'boxShadow': '0 8px 24px rgba(0,0,0,0.2)',
                    'border': f'1px solid {BORDER_COLOR}'
                }),
                
                html.Div([
                    html.Div([
                        html.H3('Analysis Results', style={'color': 'white', 'fontSize': '18px', 'fontWeight': '700', 'margin': '0 0 5px 0'}),
                        html.P('Probability of availability based on your criteria', style={'color': TEXT_GRAY, 'fontSize': '13px', 'margin': 0})
                    ]),
                    html.Div(id='output_grafico',
                             children=[
                                 html.Div([
                                     html.Div(style={'fontSize': '60px', 'marginBottom': '15px'}),
                                     html.P('Press "CALCULATE PROBABILITY"', style={'color': TEXT_GRAY, 'fontSize': '16px', 'fontWeight': '600', 'margin': '0'}),
                                     html.P('to see the results of the analysis', style={'color': TEXT_GRAY, 'fontSize': '14px', 'margin': '5px 0 0 0'})
                                 ], style={
                                     'textAlign': 'center',
                                     'padding': '60px 20px',
                                     'backgroundColor': CARD_BG,
                                     'borderRadius': '12px',
                                     'border': f'2px dashed {BORDER_COLOR}'
                                 })
                             ])
                ], style={
                    'backgroundColor': LIGHT_BG,
                    'padding': '25px',
                    'borderRadius': '16px',
                    'boxShadow': '0 8px 24px rgba(0,0,0,0.2)',
                    'border': f'1px solid {BORDER_COLOR}'
                })
            ], className='column-right')
        ], className='main-container')
    ], style={
        'backgroundColor': DARK_BG,
        'minHeight': 'calc(100vh - 140px)',
        'padding': '30px 20px'
    }),
    
    html.Div([
        html.P('© ElCanelazoDrimTim all rights reserved.', style={'textAlign': 'center', 'color': TEXT_GRAY, 'margin': 0, 'fontSize': '13px'})
    ], style={'backgroundColor': CARD_BG, 'padding': '20px', 'borderTop': f'1px solid {BORDER_COLOR}'})
], style={'backgroundColor': DARK_BG})

# CSS personalizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .main-container {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                gap: 25px;
            }
            
            .column-left {
                flex: 1;
                min-width: 0;
                display: flex;
                flex-direction: column;
                gap: 25px;
            }
            
            .column-right {
                flex: 1;
                min-width: 0;
                display: flex;
                flex-direction: column;
                gap: 25px;
            }
            
            @media (max-width: 1024px) {
                .main-container {
                    flex-direction: column;
                }
                
                .column-left, .column-right {
                    width: 100% !important;
                }
            }
            
            /* DROPDOWNS MEJORADOS - TEXTO BLANCO */
            .Select-control {
                background-color: #23262F !important;
                border: 1px solid #3A3D45 !important;
                border-radius: 8px !important;
                color: white !important;
            }
            
            .Select-menu-outer {
                background-color: #23262F !important;
                border: 1px solid #3A3D45 !important;
                border-radius: 8px !important;
                color: white !important;
                z-index: 1000 !important;
            }
            
            .Select-option {
                background-color: #23262F !important;
                color: white !important;
                padding: 12px !important;
            }
            
            .Select-option:hover {
                background-color: #2D2F36 !important;
            }
            
            .Select-value-label {
                color: white !important;
            }
            
            .Select-input > input {
                color: white !important;
            }
            
            .Select-placeholder {
                color: #B8BCC8 !important;
            }
            
            .Select--single > .Select-control .Select-value, .Select-placeholder {
                color: white !important;
            }
            
            /* Botón hover effect */
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 164, 0, 0.5) !important;
            }
            
            /* Input hover effect */
            input:focus {
                outline: none;
                border-color: #FFA400 !important;
                box-shadow: 0 0 0 3px rgba(255, 164, 0, 0.1) !important;
            }
            
            /* Checkbox hover effect */
            label:has(input[type="checkbox"]):hover {
                background-color: #2D2F36 !important;
                border-color: #FFA400 !important;
            }
            
            /* Range slider personalizado */
            .rc-slider-track {
                background-color: #FFA400 !important;
            }
            
            .rc-slider-handle {
                border-color: #FFA400 !important;
                background-color: #FFA400 !important;
            }
            
            .rc-slider-handle:active {
                box-shadow: 0 0 5px rgba(255, 164, 0, 0.5) !important;
            }
            
            /* Scrollbar personalizado */
            ::-webkit-scrollbar {
                width: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #171A21;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #3A3D45;
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #FFA400;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
# callbacks
# =============================================

@app.callback(
    Output('input_ciudad', 'options'),
    Output('input_ciudad', 'value'),
    Input('input_estado', 'value')
)
def actualizar_ciudades(estado):
    if estado is None:
        return [], None
    ciudades = ciudades_por_estado.get(estado, [])
    options = [{'label': ciudad, 'value': ciudad} for ciudad in ciudades]
    return options, ciudades[0] if ciudades else None

@app.callback(
    Output('precio_display', 'children'),
    Input('range_precio', 'value')
)
def mostrar_precio(valor):
    return f'${valor[0]} - ${valor[1]}'

@app.callback(
    Output('mapa_estados', 'figure'),
    Output('output_grafico', 'children'),
    Output('mapa_ciudad', 'figure'),
    Output('mapa_ciudad_container', 'style'),
    Input('boton_calcular', 'n_clicks'),
    State('input_estado', 'value'),
    State('input_ciudad', 'value'),
    State('input_amenities', 'value'),
    State('input_pets', 'value'),
    State('input_area', 'value'),
    State('input_habitaciones', 'value'),
    State('input_banos', 'value'),
    State('range_precio', 'value')
)
def actualizar_dashboard(n_clicks, estado, ciudad, amenities, pets, area, hab, banos, precio):
    if n_clicks == 0:
        return create_initial_map_figure(), dash.no_update, go.Figure(), {'display': 'none'}
    
    # USAR MODELO ENTRENADO
    input_data = {
        'state': estado,
        'city': ciudad,
        'bedrooms': hab,
        'bathrooms': banos,
        'square_feet': area,
        'has_gym': 'gym' in amenities,
        'has_parking': 'parking' in amenities,
        'has_pool': 'pool' in amenities,
        'has_internet': 'internet' in amenities,
        'has_elevator': 'elevator' in amenities,
        'allows_dogs': 'dogs' in pets,
        'allows_cats': 'cats' in pets
    }
    
    resultado_modelo = probability_model.predict_probability(input_data)
    probabilidad = resultado_modelo["probability"]
    lat = resultado_modelo["latitude"]
    lon = resultado_modelo["longitude"]
    city_name = resultado_modelo["city"]
    
    # VERIFICACIÓN EXTRA DE COORDENADAS
    print("=" * 60)
    print(f"VERIFICACIÓN FINAL DE COORDENADAS:")
    print(f"   Estado: {estado}")
    print(f"   Ciudad: {ciudad}") 
    print(f"   Coordenadas OBTENIDAS: Lat {lat:.6f}, Lon {lon:.6f}")
    print(f"   ¿Coordenadas por defecto?: {abs(lat - 39.8283) < 0.001 and abs(lon - (-98.5795)) < 0.001}")
    print("=" * 60)
    
    # Crear mapa de la ciudad
    mapa_ciudad = create_city_map(lat, lon, city_name, probabilidad, estado)
    
    # Actualizar mapa con estado seleccionado
    fig_mapa = create_initial_map_figure(estado)
    
    # Crear gráfico de probabilidad
    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=probabilidad,
        title={'text': 'Probability of Availability', 'font': {'color': 'white', 'size': 20}},
        number={'suffix': '%', 'font': {'color': ACCENT, 'size': 56, 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': TEXT_GRAY, 'tickfont': {'color': TEXT_GRAY, 'size': 14}},
            'bar': {'color': ACCENT, 'thickness': 0.8},
            'bgcolor': DARK_BG,
            'borderwidth': 3,
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0, 30], 'color': '#4A1C1C'},
                {'range': [30, 70], 'color': '#4A3C1C'},
                {'range': [70, 100], 'color': '#1C4A2E'}
            ],
            'threshold': {'line': {'color': 'white', 'width': 5}, 'thickness': 0.8, 'value': probabilidad}
        }
    ))
    
    fig_gauge.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font={'color': 'white'}, height=380, margin=dict(l=20, r=20, t=80, b=20))
    
    # Determinar estado de disponibilidad
    if probabilidad >= 70:
        estado_badge = html.Div('HIGH AVAILABILITY', style={'backgroundColor': '#1C4A2E', 'color': '#4ADE80', 'padding': '10px 20px', 'borderRadius': '8px', 'display': 'inline-block', 'fontWeight': '700', 'fontSize': '14px', 'border': '2px solid #4ADE80'})
    elif probabilidad >= 30:
        estado_badge = html.Div('MEDIUM AVAILABILITY', style={'backgroundColor': '#4A3C1C', 'color': ACCENT, 'padding': '10px 20px', 'borderRadius': '8px', 'display': 'inline-block', 'fontWeight': '700', 'fontSize': '14px', 'border': f'2px solid {ACCENT}'})
    else:
        estado_badge = html.Div('LOW AVAILABILITY', style={'backgroundColor': '#4A1C1C', 'color': '#EF4444', 'padding': '10px 20px', 'borderRadius': '8px', 'display': 'inline-block', 'fontWeight': '700', 'fontSize': '14px', 'border': '2px solid #EF4444'})
    
    resultado = html.Div([
        dcc.Graph(figure=fig_gauge, config={'displayModeBar': False}),
        html.Div([estado_badge], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.P('Location', style={'fontSize': '12px', 'color': TEXT_GRAY, 'margin': 0}),
                html.P(f'{ciudad}, {estado}', style={'fontSize': '16px', 'color': 'white', 'fontWeight': '600', 'margin': '5px 0 0 0'})
            ], style={'backgroundColor': CARD_BG, 'padding': '20px', 'borderRadius': '12px', 'textAlign': 'center', 'flex': '1', 'border': f'1px solid {BORDER_COLOR}'}),
            html.Div([
                html.P('Price Range', style={'fontSize': '12px', 'color': TEXT_GRAY, 'margin': 0}),
                html.P(f'${precio[0]} - ${precio[1]}', style={'fontSize': '16px', 'color': 'white', 'fontWeight': '600', 'margin': '5px 0 0 0'})
            ], style={'backgroundColor': CARD_BG, 'padding': '20px', 'borderRadius': '12px', 'textAlign': 'center', 'flex': '1', 'border': f'1px solid {BORDER_COLOR}'}),
            html.Div([
                html.P('Specifications', style={'fontSize': '12px', 'color': TEXT_GRAY, 'margin': 0}),
                html.P(f'{hab} Bedrooms | {banos} Bathrooms | {area} sq ft', style={'fontSize': '16px', 'color': 'white', 'fontWeight': '600', 'margin': '5px 0 0 0'})
            ], style={'backgroundColor': CARD_BG, 'padding': '20px', 'borderRadius': '12px', 'textAlign': 'center', 'flex': '1', 'border': f'1px solid {BORDER_COLOR}'})
        ], style={'display': 'flex', 'gap': '15px', 'marginTop': '20px', 'flexWrap': 'wrap'})
    ])
    
    return fig_mapa, resultado, mapa_ciudad, {'display': 'block'}

if __name__ == '__main__':
    app.run(debug=True)