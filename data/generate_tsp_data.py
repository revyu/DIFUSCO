import os
import numpy as np
import networkx as nx
from deap import base, creator, tools, algorithms

# Параметры
num_graphs = 1000
num_nodes = 30
pop_size = 100  # Размер популяции
num_generations = 50  # Количество поколений

# Директория для хранения файлов
script_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(script_dir, 'data')
os.makedirs(directory, exist_ok=True)

output_filename = os.path.join(directory, 'tsp_data_test.txt')

# Функция для вычисления стоимости пути (замкнутый цикл)
def evaluate_path(path, graph):
    weight = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
    weight += graph[path[-1]][path[0]]['weight']
    return weight,

def find_good_path(graph):
    # Создание типов для ГА (если уже созданы, стоит вынести за пределы цикла)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
    toolbox = base.Toolbox()
    toolbox.register("indices", np.random.permutation, num_nodes)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_path, graph=graph)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Создание начальной популяции
    population = toolbox.population(n=pop_size)
    
    # Запуск алгоритма
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, verbose=False)
    
    # Находим лучший путь
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

with open(output_filename, "w") as f:
    for i in range(num_graphs):
        # Генерируем координаты вершин
        coords = np.random.random((num_nodes, 2))
        
        # Создаём полный граф
        G = nx.complete_graph(num_nodes)
        
        # Назначаем веса на основе евклидова расстояния между координатами
        for (u, v) in G.edges():
            dist = np.sqrt((coords[u,0] - coords[v,0])**2 + (coords[u,1] - coords[v,1])**2)
            G[u][v]['weight'] = dist
        
        # Находим "хороший" путь с помощью ГА
        best_path = find_good_path(G)
        
        # Записываем в файл строку в нужном формате
        # x_1 y_1 x_2 y_2 ... x_N y_N output v_1 v_2 ... v_N v_1
        # Преобразуем координаты в строку
        coords_str = " ".join(f"{coords[n,0]} {coords[n,1]}" for n in range(num_nodes))
        
        # Преобразуем путь в строку (добавляем возвращение в начальную точку)
        path_str = " ".join(str(n+1) for n in best_path) + " " + str(best_path[0]+1)
        
        # Записываем
        f.write(f"{coords_str} output {path_str}\n")

print("Данные успешно сгенерированы и сохранены в:", output_filename)
