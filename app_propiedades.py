import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import joblib
from datetime import datetime

# =============================================
# CARGAR MODELO ENTRENADO DE TU COMPAÑERO
# =============================================

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
            print("✅ Modelo de probabilidad cargado exitosamente")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
    
    def predict_probability(self, input_data):
        if self.model is None:
            return {"error": "Modelo no disponible"}
        
        try:
            # Obtener coordenadas basadas en el estado
            state_coords = self.get_state_coordinates(input_data['state'])
            
            input_dict = {
                'bathrooms': int(input_data['bathrooms']),
                'bedrooms': int(input_data['bedrooms']),
                'square_feet': int(input_data['square_feet']),
                'latitude': state_coords['latitude'],
                'longitude': state_coords['longitude'],
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
            
            return probability * 100
            
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            return 25
    
    def normalize_to_probability(self, value):
        normalized = max(0, min(1, (value - 500) / 2000))
        return normalized
    
    def get_state_coordinates(self, state):
        state_coordinates = {
            'CA': {'latitude': 34.0522, 'longitude': -118.2437, 'city': 'Los Angeles'},
            'TX': {'latitude': 30.2672, 'longitude': -97.7431, 'city': 'Austin'},
            'NY': {'latitude': 40.7128, 'longitude': -74.0060, 'city': 'New York'},
            'FL': {'latitude': 25.7617, 'longitude': -80.1918, 'city': 'Miami'},
            'IL': {'latitude': 41.8781, 'longitude': -87.6298, 'city': 'Chicago'},
            'WA': {'latitude': 47.6062, 'longitude': -122.3321, 'city': 'Seattle'},
            'CO': {'latitude': 39.7392, 'longitude': -104.9903, 'city': 'Denver'},
            'AZ': {'latitude': 33.4484, 'longitude': -112.0740, 'city': 'Phoenix'},
            'NV': {'latitude': 36.1699, 'longitude': -115.1398, 'city': 'Las Vegas'},
            'OR': {'latitude': 45.5152, 'longitude': -122.6784, 'city': 'Portland'}
        }
        return state_coordinates.get(state, {'latitude': 39.8283, 'longitude': -98.5795, 'city': 'Unknown'})

# =============================================
# INICIALIZAR MODELO
# =============================================

probability_model = ProbabilityModel()

# =============================================
# CONFIGURACIÓN DASH (MISMO DISEÑO)
# =============================================

# Cargar datos reales desde el CSV
df = pd.read_csv('datos_entrenamiento_completo.csv')

# Extraer estados y ciudades únicos del dataset real
estados_unicos = sorted(df['state'].dropna().unique())
ciudades_por_estado = {}

for estado in estados_unicos:
    ciudades_estado = sorted(df[df['state'] == estado]['cityname'].dropna().unique())
    ciudades_por_estado[estado] = ciudades_estado

# Paleta de colores empresarial (MISMO)
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

# Función auxiliar para crear el mapa inicial (MISMA)
def create_initial_map_figure(selected_state=None):
    estados_reales = list(ciudades_por_estado.keys())
    
    z_values = [1 if estado == selected_state else 0 for estado in estados_reales]
    
    colorscale = [[0, BASE_MAP], [1, ACCENT]]
    
    fig = go.Figure(data=go.Choropleth(
        locations=estados_reales,
        z=z_values,
        locationmode='USA-states',
        colorscale=colorscale,
        showscale=False,
        marker_line_color='white',
        marker_line_width=2,
        hovertemplate='<b>%{text}</b><extra></extra>',
        text=estados_reales
    ))
    
    fig.update_geos(
        scope='usa',
        projection_type='albers usa',
        showlakes=False,
        bgcolor=CARD_BG,
        lataxis_range=[24, 50],
        lonaxis_range=[-125, -65]
    )
    
    fig.update_layout(
        geo=dict(bgcolor=CARD_BG),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        uirevision=True
    )
    
    return fig

# Calcular rangos de precios reales del dataset (MISMO)
precio_min = int(df['price'].min())
precio_max = int(df['price'].max())
precio_promedio = int(df['price'].mean())

slider_min = 0
slider_max = 5000
slider_step = 100

# Estilos (MISMOS)
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
    'marginBottom': '15px'
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

# Layout (EXACTAMENTE EL MISMO)
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1([
                html.Span(style={'fontSize': '40px'}),
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
        ], style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '30px 20px'
        })
    ], style={
        'background': f'linear-gradient(135deg, {DARK_BG} 0%, #1f2229 100%)',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
        'position': 'relative',
        'zIndex': '10'
    }),
    
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H3([
                            html.Span(style={'marginRight': '8px'}),
                            'States Map'
                        ], style={
                            'color': 'white',
                            'fontSize': '18px',
                            'fontWeight': '700',
                            'margin': '0 0 5px 0'
                        }),
                        html.P('Select a state to get started',
                               style={
                                   'color': TEXT_GRAY,
                                   'fontSize': '13px',
                                   'margin': 0
                               })
                    ], style={
                        'marginBottom': '20px'
                    }),
                    dcc.Graph(
                        id='mapa_estados',
                        figure=create_initial_map_figure(),
                        config={'displayModeBar': False},
                        style={'borderRadius': '12px', 'overflow': 'hidden'}
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
                        html.H3([
                            html.Span(style={'marginRight': '8px'}),
                            'Price Range'
                        ], style={
                            'color': 'white',
                            'fontSize': '18px',
                            'fontWeight': '700',
                            'margin': '0 0 5px 0'
                        }),
                        html.P(f'Real data range: ${precio_min} - ${precio_max}',
                               style={
                                   'color': TEXT_GRAY,
                                   'fontSize': '13px',
                                   'margin': 0
                               })
                    ], style={
                        'marginBottom': '25px'
                    }),
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
                    html.Div(id='precio_display',
                             style={
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
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H3([
                            html.Span(style={'marginRight': '8px'}),
                            'Search Settings'
                        ], style={
                            'color': 'white',
                            'fontSize': '18px',
                            'fontWeight': '700',
                            'margin': '0 0 5px 0'
                        }),
                        html.P('Complete the details of the desired property',
                               style={
                                   'color': TEXT_GRAY,
                                   'fontSize': '13px',
                                   'margin': 0
                               })
                    ], style={
                        'marginBottom': '25px'
                    }),
                    
                    html.Div([
                        html.Label('Estado', style=label_style),
                        dcc.Dropdown(
                            id='input_estado',
                            options=[{'label': f'{estado}', 'value': estado} for estado in ciudades_por_estado.keys()],
                            value=list(ciudades_por_estado.keys())[0] if ciudades_por_estado else None,
                            style=dropdown_style,
                            placeholder='Select a state',
                            className='custom-dropdown'
                        ),
                    ]),
                    
                    html.Div([
                        html.Label('Ciudad', style=label_style),
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
                    
                    html.Button([
                        html.Span(style={'marginRight': '10px', 'fontSize': '20px'}),
                        'CALCULATE PROBABILITY'
                    ],
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
                        html.H3([
                            html.Span(style={'marginRight': '8px'}),
                            'Analysis Results'
                        ], style={
                            'color': 'white',
                            'fontSize': '18px',
                            'fontWeight': '700',
                            'margin': '0 0 5px 0'
                        }),
                        html.P('Probability of availability based on your criteria',
                               style={
                                   'color': TEXT_GRAY,
                                   'fontSize': '13px',
                                   'margin': 0
                               })
                    ], style={
                        'marginBottom': '20px'
                    }),
                    html.Div(id='output_grafico',
                             children=[
                                 html.Div([
                                     html.Div(style={'fontSize': '60px', 'marginBottom': '15px'}),
                                     html.P('Press "CALCULATE PROBABILITY"',
                                            style={
                                                'color': TEXT_GRAY,
                                                'fontSize': '16px',
                                                'fontWeight': '600',
                                                'margin': '0'
                                            }),
                                     html.P('to see the results of the analysis',
                                            style={
                                                'color': TEXT_GRAY,
                                                'fontSize': '14px',
                                                'margin': '5px 0 0 0'
                                            })
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
        html.P([
            '© ElCanelazoDrimTim all rights reserved. ',
            html.Span( style={'color': ACCENT, 'fontWeight': '600'})
        ], style={
            'textAlign': 'center',
            'color': TEXT_GRAY,
            'margin': 0,
            'fontSize': '13px'
        })
    ], style={
        'backgroundColor': CARD_BG,
        'padding': '20px',
        'borderTop': f'1px solid {BORDER_COLOR}'
    })
], style={
    'backgroundColor': DARK_BG,
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
})

# CSS personalizado (MISMO)
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
            }
            
            .column-right {
                flex: 1;
                min-width: 0;
            }
            
            @media (max-width: 1024px) {
                .main-container {
                    flex-direction: column;
                }
                
                .column-left, .column-right {
                    width: 100% !important;
                }
            }
            
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
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 164, 0, 0.5) !important;
            }
            
            input:focus {
                outline: none;
                border-color: #FFA400 !important;
                box-shadow: 0 0 0 3px rgba(255, 164, 0, 0.1) !important;
            }
            
            label:has(input[type="checkbox"]):hover {
                background-color: #2D2F36 !important;
                border-color: #FFA400 !important;
            }
            
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

# =============================================
# CALLBACKS (MODIFICADO PARA USAR MODELO ENTRENADO)
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
    options = [{'label': f'{ciudad}', 'value': ciudad} for ciudad in ciudades]
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
        return create_initial_map_figure(), dash.no_update
    
    if not estado or not ciudad:
        return create_initial_map_figure(), html.Div([
            html.Div([
                html.Div(style={'fontSize': '40px', 'color': '#EF4444', 'marginBottom': '15px'}),
                html.P('Please select both state and city',
                       style={
                           'color': TEXT_GRAY,
                           'fontSize': '16px',
                           'fontWeight': '600',
                           'margin': '0'
                       })
            ], style={
                'textAlign': 'center',
                'padding': '60px 20px',
                'backgroundColor': CARD_BG,
                'borderRadius': '12px',
                'border': f'2px dashed #EF4444'
            })
        ])
    
    if area is None or hab is None or banos is None:
        return create_initial_map_figure(estado), html.Div([
            html.Div([
                html.Div(style={'fontSize': '40px', 'color': '#EF4444', 'marginBottom': '15px'}),
                html.P('Please fill all property specifications',
                       style={
                           'color': TEXT_GRAY,
                           'fontSize': '16px',
                           'fontWeight': '600',
                           'margin': '0'
                       })
            ], style={
                'textAlign': 'center',
                'padding': '60px 20px',
                'backgroundColor': CARD_BG,
                'borderRadius': '12px',
                'border': f'2px dashed #EF4444'
            })
        ])
    
    # USAR MODELO ENTRENADO EN LUGAR DE CÁLCULO MANUAL
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
    
    probabilidad = probability_model.predict_probability(input_data)
    
    # Actualizar mapa con estado seleccionado
    fig_mapa = create_initial_map_figure(estado)
    
    # Crear gráfico de probabilidad (MISMA VISUALIZACIÓN)
    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=probabilidad,
        title={
            'text': 'Probability of Availability',
            'font': {'color': 'white', 'size': 20, 'family': 'Arial, sans-serif'}
        },
        number={
            'suffix': '%',
            'font': {'color': ACCENT, 'size': 56, 'family': 'Arial, sans-serif', 'weight': 'bold'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': TEXT_GRAY,
                'tickfont': {'color': TEXT_GRAY, 'size': 14}
            },
            'bar': {'color': ACCENT, 'thickness': 0.8},
            'bgcolor': DARK_BG,
            'borderwidth': 3,
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0, 30], 'color': '#4A1C1C'},
                {'range': [30, 70], 'color': '#4A3C1C'},
                {'range': [70, 100], 'color': '#1C4A2E'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 5},
                'thickness': 0.8,
                'value': probabilidad
            }
        }
    ))
    
    fig_gauge.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={'color': 'white', 'family': 'Arial, sans-serif'},
        height=380,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    # Determinar estado de disponibilidad (MISMA LÓGICA)
    if probabilidad >= 70:
        estado_badge = html.Div([
            html.Span(style={'marginRight': '8px', 'fontSize': '20px'}),
            'HIGH AVAILABILITY'
        ], style={
            'backgroundColor': '#1C4A2E',
            'color': '#4ADE80',
            'padding': '10px 20px',
            'borderRadius': '8px',
            'display': 'inline-block',
            'fontWeight': '700',
            'fontSize': '14px',
            'border': '2px solid #4ADE80'
        })
    elif probabilidad >= 30:
        estado_badge = html.Div([
            html.Span(style={'marginRight': '8px', 'fontSize': '20px'}),
            'MEDIUM AVAILABILITY'
        ], style={
            'backgroundColor': '#4A3C1C',
            'color': ACCENT,
            'padding': '10px 20px',
            'borderRadius': '8px',
            'display': 'inline-block',
            'fontWeight': '700',
            'fontSize': '14px',
            'border': f'2px solid {ACCENT}'
        })
    else:
        estado_badge = html.Div([
            html.Span(style={'marginRight': '8px', 'fontSize': '20px'}),
            'LOW AVAILABILITY'
        ], style={
            'backgroundColor': '#4A1C1C',
            'color': '#EF4444',
            'padding': '10px 20px',
            'borderRadius': '8px',
            'display': 'inline-block',
            'fontWeight': '700',
            'fontSize': '14px',
            'border': '2px solid #EF4444'
        })
    
    resultado = html.Div([
        dcc.Graph(figure=fig_gauge, config={'displayModeBar': False}),
        html.Div([
            estado_badge
        ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.Div(style={'fontSize': '24px', 'marginBottom': '5px'}),
                html.P('Location', style={'fontSize': '12px', 'color': TEXT_GRAY, 'margin': 0}),
                html.P(f'{ciudad}, {estado}', style={'fontSize': '16px', 'color': 'white', 'fontWeight': '600', 'margin': '5px 0 0 0'})
            ], style={
                'backgroundColor': CARD_BG,
                'padding': '20px',
                'borderRadius': '12px',
                'textAlign': 'center',
                'flex': '1',
                'border': f'1px solid {BORDER_COLOR}'
            }),
            html.Div([
                html.Div(style={'fontSize': '24px', 'marginBottom': '5px'}),
                html.P('Price Range', style={'fontSize': '12px', 'color': TEXT_GRAY, 'margin': 0}),
                html.P(f'${precio[0]} - ${precio[1]}', style={'fontSize': '16px', 'color': 'white', 'fontWeight': '600', 'margin': '5px 0 0 0'})
            ], style={
                'backgroundColor': CARD_BG,
                'padding': '20px',
                'borderRadius': '12px',
                'textAlign': 'center',
                'flex': '1',
                'border': f'1px solid {BORDER_COLOR}'
            }),
            html.Div([
                html.Div(style={'fontSize': '24px', 'marginBottom': '5px'}),
                html.P('Specifications', style={'fontSize': '12px', 'color': TEXT_GRAY, 'margin': 0}),
                html.P(f'{hab} Bedrooms | {banos} Bathrooms | {area} sq ft', style={'fontSize': '16px', 'color': 'white', 'fontWeight': '600', 'margin': '5px 0 0 0'})
            ], style={
                'backgroundColor': CARD_BG,
                'padding': '20px',
                'borderRadius': '12px',
                'textAlign': 'center',
                'flex': '1',
                'border': f'1px solid {BORDER_COLOR}'
            })
        ], style={
            'display': 'flex',
            'gap': '15px',
            'marginTop': '20px',
            'flexWrap': 'wrap'
        })
    ])
    
    return fig_mapa, resultado

if __name__ == '__main__':
    app.run(debug=True)