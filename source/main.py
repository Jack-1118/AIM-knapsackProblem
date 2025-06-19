import matplotlib.pyplot as plt
from bag import Bag
from dna import Dna
from input_handler import get_params
from population import Population
import pandas as pd

if __name__ == "__main__":
    """Getting parameters from the given text file. 
    """
    params, file_name = get_params()
    iter_number = params.get('iter_number')
    #print(params)


    """Genetic algorithm:
    """
    pop = Population(params)
    pop.initialize_population()
    summary = []
    for i in range(iter_number):
        print('Generation', i, '\n', pop)
        summary.append(pop.pop_summary())
        parents = []
        """
        for _ in range(int(params.get('pop_size') / 2)):
            pars = pop.select_parents(params.get('tournament_size'))
            parents.append(pars)
        """
        for _ in range(int(params.get('pop_size'))):
            par = pop.select_parent(params.get('tournament_size'))
            parents.append(par)

        children = pop.recombine(parents, single=True)
        pop.mutate_children(children)
        mating_pool = pop.pop.copy()
        mating_pool += children

        pop.pop = pop.survivor_select(mating_pool)
    
    print('Final Population:', pop)

    # Print the best solution and its fitness only
    best_individual = max(pop.pop, key=lambda x: x.fitness())

    # Prepare selected items with 'Index' starting from 1
    selected_items = []
    for idx, (bit, w, v) in enumerate(zip(best_individual.bits, params['item_weights'], params['item_values'])):
        if bit == '1':
            selected_items.append({'Index': idx + 1, 'Weight': w, 'Value': v})

    # Add total weight and value as a summary row
    df = pd.DataFrame(selected_items)
    if not df.empty:
        total_weight = df['Weight'].sum()
        total_value = df['Value'].sum()
        summary_row = pd.DataFrame([{'Index': 'Total', 'Weight': total_weight, 'Value': total_value}])
        df = pd.concat([df, summary_row], ignore_index=True)

    """Creating the summaries for graph.
    """
    best_vals = [s[0] for s in summary]
    avg_vals = [s[1] for s in summary]
    worst_vals = [s[2] for s in summary]

    """Plotting the graph
    """
    range_ = range(0, iter_number)
    plt.plot(range_, best_vals, 'go--', label='Best')
    plt.plot(range_, avg_vals, 'rs--', label='Average')
    plt.plot(range_, worst_vals, 'bo--', label='Worst')
    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    title = 'Fitness values for ' 
    title += str(file_name)
    plt.title(title)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(file_name)

    # Display the items in the best solution as a table in a pop-up window
    if not df.empty:
        fig_table, ax_table = plt.subplots(figsize=(6, len(df)*0.5+1))
        ax_table.axis('off')
        table = ax_table.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            colLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        plt.title('Items in Best Solution')

    # Show all figures at once
    plt.show()
