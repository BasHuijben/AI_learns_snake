import os
from datetime import datetime
import re
import pickle
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from functions.train_ai import start_genetic_algorithm
from functions.generate_dashboard_layout import generate_layout
from default import output_path, file_name_intermediate_results

app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])
app.layout = generate_layout()


# Callbacks for tab 1
@app.callback(Output('train_ai_button', 'disabled'),
              [Input('train_ai_button_triggered', 'children'),
               Input('output_folder', 'children'),
               Input('number_of_generations', 'value'),
               Input('population_size', 'value'),
               Input('survival_perc', 'value'),
               Input('parent_perc', 'value'),
               Input('mutation_perc', 'value')])
def train_ai(button_triggered, output_folder, number_of_generations, population_size, survival_perc, parent_perc,
             mutation_perc):
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    # if the train button is triggered, start the genetic algorithm
    if button_triggered and context == 'train_ai_button_triggered':
        start_genetic_algorithm(output_folder, number_of_generations, population_size, survival_perc, parent_perc,
                                mutation_perc)
    return False


@app.callback([Output('train_ai_button_triggered', 'children'),
               Output('output_folder', 'children')],
              Input('train_ai_button', 'n_clicks'))
def train_ai_triggered(_):
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M")
    output_folder = os.path.join(output_path, start_time_str)
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if context == 'train_ai_button':
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        return True, output_folder
    return False, None


@app.callback(Output('display', 'figure'),
              [Input('interval', 'n_intervals'),
               Input('output_folder', 'children'),
               Input('metric_drop_down', 'value')])
def update_graph(_, output_folder, metric):
    if output_folder is not None:
        intermediate_results_path = os.path.join(output_folder, file_name_intermediate_results)
        if os.path.exists(intermediate_results_path):

            # read intermediate results
            df = pd.read_csv(intermediate_results_path)

            # depending on the metric select the correct columns
            if metric == 'score':
                population = 'population_score'
                best_snake = 'best_score'
                y_label = "Number of found apples (n)"
            elif metric == 'fitness':
                population = 'population_fitness'
                best_snake = 'best_fitness'
                y_label = "Fitness"
            else:
                raise ValueError("Metric %s is not supported" % metric)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['generation'],
                                     y=df[population],
                                     name='Population %s' % metric,
                                     mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df['generation'],
                                     y=df[best_snake],
                                     name='Best snake %s' % metric,
                                     mode='lines+markers'))
            fig.update_layout(title="Progress of Snake AI",
                              xaxis_title="Generation (n)",
                              yaxis_title=y_label)
            return fig
    return go.Figure()

# Callbacks for tab 2
@app.callback(Output('session_selection', 'options'),
              Input('session_interval', 'n_intervals'))
def set_session_selection(_):
    sessions = sorted(os.listdir(output_path), reverse=True)
    options = []
    for session in sessions:
        if os.path.isdir(os.path.join(output_path, session)):
            options.append({'label': 'Trained AI at %s' % session,
                            'value': session})
    return options

@app.callback(Output('generation_selection', 'options'),
              Input('session_selection', 'value'))
def set_generation_selection(session_name):
    options = []
    if session_name is not None:
        path = os.path.join(output_path, session_name)
        if os.path.isdir(path):
            generations = os.listdir(path)
            for generation in generations:
                if os.path.isfile(os.path.join(path, generation)) and os.path.splitext(generation)[1] == '.obj':
                    try:
                        match = re.search(r"^best_snake_generatie-(\d+)_score-(\d+).obj$", generation).groups()

                        options.append({'label': 'Generation %s, score=%s' % (match[0], match[1]),
                                        'value': generation,
                                        'generation': int(match[0])})
                    except:
                        pass
            # sort generations from high to low
            options = sorted(options, key=lambda k: k['generation'], reverse=True)

            # remove key='generation' because only the keys label and value are excepted by drop down component
            for option in options:
                option.pop('generation')
            return options
    return options


@app.callback(Output('snake_location', 'children'),
              [Input('session_selection', 'value'),
               Input('generation_selection', 'value'),
               Input('play_ai_button', 'n_clicks'),
               Input('snake_location', 'children')])
def ai_plays_snake(session, generation, _, snake):
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if context == 'play_ai_button':
        snake_location = os.path.join(output_path, session, generation)
        with open(snake_location, 'rb') as file:
            snake = pickle.load(file)
            snake.reset_snake()
        with open(snake_location, 'wb') as file:
            pickle.dump(snake, file)

        return snake_location
    return snake


@app.callback(Output('play_snake', 'figure'),
              [Input('snake_interval', 'n_intervals'),
               Input('snake_location', 'children'),
               Input('play_snake', 'figure')])
def update_graph(_, snake_location, fig):

    if snake_location is not None:
        with open(snake_location, 'rb') as file:
            snake = pickle.load(file)
        if snake.alive:
            fig = go.Figure()

            # plot board
            fig.add_trace(go.Scatter(x=[0, 0, snake.grid_size, snake.grid_size, 0],
                                     y=[snake.grid_size, 0, 0, snake.grid_size, snake.grid_size],
                                     line={'color': 'Black'},
                                     mode='lines'))
            # plot snake head
            fig.add_trace(go.Scatter(x=[snake.snake_head[0, 0], snake.snake_head[0, 0], snake.snake_head[0, 0]+1,
                                        snake.snake_head[0, 0]+1, snake.snake_head[0, 0]],
                                     y=[snake.snake_head[0, 1]+1, snake.snake_head[0, 1], snake.snake_head[0, 1],
                                        snake.snake_head[0, 1]+1, snake.snake_head[0, 1]+1],
                                     fill="toself",
                                     line={'color': 'Black'},
                                     mode='lines'))

            # plot snake body
            for body_part in snake.snake_body:
                fig.add_trace(go.Scatter(x=[body_part[0], body_part[0], body_part[0]+1, body_part[0]+1, body_part[0]],
                                         y=[body_part[1]+1, body_part[1], body_part[1], body_part[1]+1, body_part[1]+1],
                                         fill="toself",
                                         line={'color': 'Red'},
                                         mode='lines'))

            # plot apple
            fig.add_trace(go.Scatter(x=[snake.apple[0, 0], snake.apple[0, 0], snake.apple[0, 0]+1,
                                        snake.apple[0, 0]+1, snake.apple[0, 0]],
                                     y=[snake.apple[0, 1]+1, snake.apple[0, 1], snake.apple[0, 1],
                                        snake.apple[0, 1]+1, snake.apple[0, 1]+1],
                                     fill="toself",
                                     line={'color': 'Green'},
                                     mode='lines'))

            # modify of style of the figure
            fig.update_layout(title={'text': 'Apples found = %i' % snake.total_apples_found,
                                     'y': 0.9,
                                     'x': 0.5,
                                     'xanchor': 'center',
                                     'yanchor': 'top'},
                              showlegend=False,
                              hovermode=False,
                              width=800,
                              height=800,
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(autorange=True,
                                         showgrid=False,
                                         ticks='',
                                         showticklabels=False,
                                         range=[0, snake.grid_size]),
                              yaxis=dict(autorange=True,
                                         showgrid=False,
                                         ticks='',
                                         showticklabels=False,
                                         range=[0, snake.grid_size]))

            # after drawing the snake, let the snake take another step
            snake.snake_move()
            with open(snake_location, 'wb') as file:
                pickle.dump(snake, file)

            return fig
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
