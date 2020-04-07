import numpy as np
import initialisation
import task
import ordre
import data_loading as dtld
import time_personalized
from pathlib import Path
import analysis
import printgraph
from time import time
import matplotlib.pyplot as plt
from statistics import variance

# PARAMETERS

# graph to use
data_folder = Path("../graphs") 
path_graph = data_folder / "mediumRandom.json"

# sizes
n_population = 10
n_cores = 4
# generation : sum must be equal to n_population
# n_selected is the number of best individuals kept between each iteration, same idea for n_mutated and n_crossed
n_selected = 3
n_mutated = 4
n_crossed = 3
# genetics --> adapt the blocs size and the mutation numbers to the number of tasks
mutations_prob = 0.6
nb_mut_max = 10
crossover_bloc_size = (1, 7) # must be inferior to n_tasks
# execution
epochs = 10
# logs during the execution?
verbose = True
time_analytics = True
colored_graph_displaying = True
verify_legality = False
graph_evolution = True







# MAIN CODE

def main_genetics(path_graph, n_population, n_cores, n_selected, n_mutated, n_crossed, mutation_prob, nb_mut_max, crossover_bloc_size, epochs, verbose=True, time_analytics=True, colored_graph_displaying=True, verify_legality=False, graph_evolution=False):
    '''
    Main call function that starts the genetic algorithm
    '''
    # load datas
    tasks_dict = dtld.loadTasks(path_graph)
    n_tasks = len(tasks_dict.keys())
    optimal_time = dtld.ideal_time(tasks_dict)

    # initial population
    population = initialisation.population_initiale(tasks_dict, n_population)
    scores = ordre.population_eval(population, n_cores, tasks_dict, optimal_time)

    # time_analytics
    ana_ordres = []
    ana_scores = []
    ana_means = []

    # graph evolution
    graph_evo_best = []

    best_10 = 0
    best_30 = 0

    # execution
    for epoch in range(epochs):
        if verbose:
            print(f'\n__________epoch n{epoch}_________')
        # selection of the bests
        if verbose:
            best_ordres = ordre.selection_nbest(population, n_selected, scores, verbose=True)
        if epoch == 9:
            best_ordres = ordre.selection_nbest(population, n_selected, scores)
            best_10 = ordre.population_eval([ordre.selection_nbest(population, n_selected, scores)[0]], n_cores, tasks_dict, optimal_time)[0]
        if epoch == 29:
            best_ordres = ordre.selection_nbest(population, n_selected, scores)
            best_30 = ordre.population_eval([ordre.selection_nbest(population, n_selected, scores)[0]], n_cores, tasks_dict, optimal_time)[0]
        else:
            best_ordres = ordre.selection_nbest(population, n_selected, scores)
        if graph_evolution:
            graph_evo_best.append(best_ordres[0])
        # mutations
        mutated_ordres = []
        for i in range(n_mutated):
            add_index = np.random.randint(0, n_selected, 1)[0]
            mutated_ordres.append(best_ordres[add_index].mutation_multiple_out(nb_mut_max, mutation_prob, tasks_dict))
        # crossovers
        cross_ordres = []
        for i in range(n_crossed):
            parents = np.random.randint(0, n_selected, 2)
            bloc_size = np.random.randint(crossover_bloc_size[0], crossover_bloc_size[1])
            cross_ordres.append(ordre.crossover_2_parents(best_ordres[parents[0]], best_ordres[parents[1]], bloc_size))
        # evaluations
        population = best_ordres + mutated_ordres + cross_ordres
        scores = ordre.population_eval(population, n_cores, tasks_dict, optimal_time)
        # legality checking
        if verify_legality:
            for ind in population:
                if not ind.isLegal(tasks_dict, n_tasks):
                    return f'ERROR !!! illegal order thrown at ' + str(ind)
        # time_analytics
        if time_analytics:
            ana_ordres.append(best_ordres[0].ordre)
            ana_scores.append(scores[0])
            ana_means.append(ordre.mean(scores))

    # log results to the console
    bar = '\n_______________________'
    print('\n' + bar + '\n____________RESULTS__________' + bar + '\n')
    best_result = ordre.selection_nbest(population, 1, scores)[0]
    print(best_result)
    best_score = ordre.population_eval([best_result], n_cores, tasks_dict, optimal_time)[0]
    print('the best ordre has a score of : ', best_score)
    print(bar + '\n\n')

    # time_analytics score printing
    if time_analytics:
        analysis.performance_evaluation(ana_scores, ana_means)

    # graph printing
    if colored_graph_displaying:
        printgraph.print_color_graph(tasks_dict, best_result)

    # graph evolution
    if graph_evolution:
        pos = printgraph.getpos(tasks_dict)
        for i in range(epochs):
            printgraph.print_color_graph(tasks_dict, graph_evo_best[i], pos=pos, title=f'best at epoch {i}')
    return best_result, best_score, best_10, best_30


def repetability_analysis():
    '''
    analysis of the repetability of the algorithm, convergence speed and computational time for different population sizes
    '''
    # graph to use
    data_folder = Path("../graphs")
    path_graph = data_folder / "smallComplex.json"
    # sizes
    n_population = 10
    n_cores = 4
    # generation : sum must be equal to n_population
    # n_selected is the number of best individuals kept between each iteration, same idea for n_mutated and n_crossed
    n_selected = 3
    n_mutated = 4
    n_crossed = 3
    # genetics --> adapt the blocs size and the mutation numbers to the number of tasks
    mutations_prob = 0.6
    nb_mut_max = 800
    crossover_bloc_size = (20, 200)  # must be inferior to n_tasks
    # execution
    epochs = 50
    # logs during the execution?
    verbose = False
    time_analytics = False
    colored_graph_displaying = False
    verify_legality = False
    graph_evolution = False


    # giving times
    res_scores_50 = [[] for _ in range(10)]
    res_scores_10 = [[] for _ in range(10)]
    res_scores_30 = [[] for _ in range(10)]
    times = [[] for _ in range(10)]

    print('begining\n')

    for i in range(8):
        for j in range(10):
            start_time = time()
            best_result = main_genetics(path_graph, n_population * (i + 1), n_cores, n_selected * (i + 1), n_mutated * (i + 1), n_crossed * (i + 1), mutations_prob, nb_mut_max, crossover_bloc_size, epochs, verbose=verbose, time_analytics=time_analytics, colored_graph_displaying=colored_graph_displaying, verify_legality=verify_legality, graph_evolution=graph_evolution)
            end_time = time()
            times[i].append(end_time-start_time)
            res_scores_50[i].append(best_result[1])
            res_scores_10[i].append(best_result[2])
            res_scores_30[i].append(best_result[3])
            print(f'\npop size {i}\niteration {j}\n')

    print('scores at 10 epochs', res_scores_10)
    print('scores at 30 epochs', res_scores_30)
    print('scores at 50 epochs', res_scores_50)
    print('times to compute 50 epochs', times)

    return res_scores_10, res_scores_30, res_scores_50, times

#results for the smallComplex graph (1000 tasks)

res10 = [[0.6647589112840813, 0.6853019753279981, 0.6718336836271115, 0.6579191479590554, 0.6825661135339844, 0.6597544719275579, 0.686021196397437, 0.6946495251767584, 0.6710312277073254, 0.6864032674516805], [0.6470202056691388, 0.67140667446088, 0.6503708320179233, 0.6600575913052571, 0.6604353685151105, 0.6540371540987759, 0.6712360133534356, 0.6636342996585851, 0.6527268657151755, 0.6799432105787193], [0.6533415427711293, 0.6508679768213479, 0.6640440544849096, 0.6499546855097162, 0.6767953403303117, 0.6489921024864169, 0.6637539062733149, 0.6601194785783511, 0.6640611765095321, 0.6603350265789634], [0.6445381373952768, 0.6493827569612054, 0.6599208797388445, 0.6310890164718033, 0.6536339147331487, 0.6487915829411695, 0.65185193303131, 0.6644943020564902, 0.6559644051781623, 0.6578422429011841], [0.6454538018455573, 0.6666390645245797, 0.6420499335336309, 0.6590203547180384, 0.6634913543349841, 0.6441477432828222, 0.665772661908322, 0.6574154215372916, 0.665484624338925, 0.6430358253881567], [0.6512244491362982, 0.6498052289936531, 0.6480061466083666, 0.6599615304087949, 0.6583481738663151, 0.646254078831386, 0.6641272978717263, 0.6596755629333633, 0.6774442315578661, 0.6517601979910002], [0.6648230116370124, 0.6606487183747083, 0.6578978238570949, 0.6534759025400678, 0.6614190280827417, 0.6525988933598845, 0.6616559919523668, 0.6514293863047502, 0.6558811255113481, 0.6453150756722341], [0.6534396438839118, 0.6328945502942622, 0.652600837540918, 0.6516106411714151, 0.6457673912036628, 0.6420327453513661, 0.6592764189391995, 0.6435884764533799, 0.6568946947356022, 0.6508083538469107], [], []]
res30 = [[0.6450333209447279, 0.6673882165918166, 0.6602624324384219, 0.6320984433740091, 0.6593981319278919, 0.6544959317379644, 0.6719808352962853, 0.6873677413030761, 0.6497389817185268, 0.6581432089543358], [0.6239930288668047, 0.6437849262365738, 0.6266330623831624, 0.6425684749990108, 0.6386587504160655, 0.6332850468443565, 0.645289481201176, 0.6582584299576235, 0.6451719809604088, 0.658190229964994], [0.643893721412099, 0.6376728201474251, 0.6350927233207753, 0.6291831855296646, 0.6550189527159556, 0.6217160322107966, 0.6435052522736204, 0.6423251002404455, 0.6491800222020032, 0.6300561825689666], [0.6313707434557823, 0.6235576646305996, 0.6363495840730125, 0.6056609617613988, 0.6396703028992914, 0.6370539474164858, 0.6471525166852867, 0.6334280156432497, 0.6386651122203033, 0.636441348990967], [0.6247592154598471, 0.624025042763273, 0.6295781807994278, 0.6325188773242538, 0.6428565189707602, 0.6241916277063111, 0.6324761864379829, 0.6314753792363192, 0.6274525165487566, 0.635444854840751], [0.6343136231828006, 0.6301565117004089, 0.6301821083055845, 0.6355856553762744, 0.6369536310897486, 0.637158553319378, 0.64662317445129, 0.6392008354655956, 0.6468515911803674, 0.6394100536737324], [0.630559672103689, 0.6278536089921889, 0.6258155588009542, 0.6162376693832579, 0.6371775939156241, 0.6229793273272661, 0.6282741581847782, 0.6227509511464211, 0.624522260126692, 0.6152538369521101], [0.6314859665931876, 0.619204152449716, 0.6184721266753368, 0.6258028437289485, 0.6182523381832716, 0.6209626396521069, 0.622031751418157, 0.6304744205123141, 0.6275314511523611, 0.6273820757327626], [], []]
res50 = [[0.6417767408854949, 0.6641016479136135, 0.6559471551064902, 0.6287030368379309, 0.6464269466163148, 0.6538068059933699, 0.6584378068008971, 0.6839404254422865, 0.6497389817185268, 0.6388914033683608], [0.6214920053613961, 0.6437849262365738, 0.6266330623831624, 0.6403574673899124, 0.6214641807375609, 0.6263447921949592, 0.645289481201176, 0.6450204650209679, 0.6276788076968141, 0.6468878370318185], [0.6396916334036045, 0.6376728201474251, 0.6263875214953449, 0.6225610104215717, 0.645024846364175, 0.6140672228111754, 0.6309759722683554, 0.6334301369560349, 0.6429441138231393, 0.6249789698060324], [0.6311487482862324, 0.6235576646305996, 0.6312511496457573, 0.6056609617613988, 0.6329691782487716, 0.6370539474164858, 0.6391729745617629, 0.6295567371868878, 0.6270703964098108, 0.6220510075602697], [0.6247592154598471, 0.6158470682614066, 0.6190078307135716, 0.6211141513233129, 0.629701812359744, 0.6211184622406429, 0.6246247340462117, 0.6310933465961905, 0.6170508065610816, 0.6336393060794696], [0.6343136231828006, 0.62869249429753, 0.6285217584959737, 0.6258603368541305, 0.6306921858609986, 0.637158553319378, 0.6227382830250001, 0.6311167984132888, 0.6380398414051289, 0.6164617794632405], [0.6304232913254875, 0.6184016303722881, 0.6189118317065165, 0.610578053823172, 0.6354383372459362, 0.6148419117282984, 0.6200363387602559, 0.6147458849777161, 0.623339991048967, 0.6061412833166491], [0.6292791973414245, 0.6169248786903461, 0.6142829586141272, 0.6258028437289485, 0.604881974731651, 0.6200940965160067, 0.6165023063543764, 0.6304744205123141, 0.6091866326949522, 0.6135294358742704], [], []]
times = [[13.014105081558228, 13.280520915985107, 13.185956954956055, 13.043120861053467, 13.263588666915894, 12.54446029663086, 13.441992998123169, 12.780823230743408, 13.666486263275146, 12.885562896728516], [26.244823455810547, 25.762122631072998, 26.680274963378906, 29.70964217185974, 30.620828866958618, 29.38161063194275, 29.27516770362854, 29.310898065567017, 29.793972969055176, 25.640472173690796], [43.283586740493774, 44.890146017074585, 42.370129108428955, 41.96259093284607, 71.17231440544128, 48.55425000190735, 46.48470091819763, 46.093382358551025, 40.26913285255432, 46.316471576690674], [68.57762956619263, 76.11464142799377, 68.26918125152588, 64.53353762626648, 64.80702424049377, 63.8687481880188, 60.47804522514343, 67.97280025482178, 74.88908672332764, 68.21117210388184], [91.02281594276428, 79.92452907562256, 73.72135639190674, 83.22384810447693, 91.43091583251953, 100.08986973762512, 83.77049779891968, 89.54529547691345, 88.34121417999268, 86.99205708503723], [87.89744424819946, 85.79452776908875, 82.7189576625824, 86.094003200531, 83.86472153663635, 90.87577319145203, 94.66162824630737, 85.27018857002258, 92.04436254501343, 80.65098905563354], [100.50489807128906, 100.12672138214111, 99.9216423034668, 113.12064862251282, 97.51923251152039, 101.62065601348877, 95.87338328361511, 94.63753390312195, 93.04324650764465, 95.79776000976562], [124.03403544425964, 116.73437905311584, 113.30650973320007, 114.20740580558777, 112.29059386253357, 104.68846869468689, 106.16105842590332, 104.4876971244812, 103.66179370880127, 105.08798170089722], [], []]

def exploitation_repetability_analysis(res10, res30, res50, time50):
    '''
    display different graphs analysis
    '''
    #scores
    x_util = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    x_pop = [10 + (k/2) for k in x_util] + [30 + (k/2) for k in x_util] + [50 + (k/2) for k in x_util]
    x = []
    for i in range(len(x_pop)):
        for _ in range(8):
            x.append(x_pop[i])
    y = []
    for i in range(8):
        for j in range(10):
            y.append(res10[i][j])
    for i in range(8):
        for j in range(10):
            y.append(res30[i][j])
    for i in range(8):
        for j in range(10):
            y.append(res50[i][j])
    plt.figure()
    plt.scatter(x, y, s=[3 for _ in range(240)])
    plt.show()

    #temps d'executions
    x2 = []
    y2 = []
    for i in range(8):
        for j in range(10):
            x2.append((i+1)*10)
            y2.append(time50[i][j])
    plt.figure()
    plt.scatter(x2, y2, s=[3 for _ in range(240)])
    plt.show()

    #variances
    variances = [variance(res50[i]) for i in range(8)]
    x3 = [(i+1)*10 for i in range(8)]
    plt.figure()
    plt.plot(x3, variances)
    plt.show()


if __name__ == "__main__" :
    # uncomment this section to start the algorithm without parallelisation
    # the parameters are specified at the top of this file
    '''
    start_time = time()
    best_result = main_genetics(path_graph, n_population, n_cores, n_selected, n_mutated, n_crossed, mutations_prob, nb_mut_max, crossover_bloc_size, epochs, verbose=verbose, time_analytics=time_analytics, colored_graph_displaying=colored_graph_displaying, verify_legality=verify_legality, graph_evolution=graph_evolution)
    end_time = time()
    print('Total Time : {0} s'.format(end_time-start_time))
    '''

    # uncomment this section to see the exploitation of the results
    '''
    exploitation_repetability_analysis(res10, res30, res50, times)
    '''