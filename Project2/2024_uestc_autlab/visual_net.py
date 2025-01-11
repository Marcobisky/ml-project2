import graphviz

def create_condensed_network_visualization():
    dot = graphviz.Digraph(
        'Condensed AlexNet Architecture',
        node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontname': 'Arial'},
        edge_attr={'fontname': 'Arial'},
        graph_attr={'rankdir': 'LR', 'splines': 'ortho'}
    )
    
    # Color scheme
    colors = {
        'input': '#E6F3FF',
        'feature': '#FFE6E6',
        'pool': '#E6FFE6',
        'fc': '#FFE6FF',
        'output': '#FFFFD4'
    }

    # Add grouped feature extraction
    dot.node('input', 'Input\n192×192×3', fillcolor=colors['input'])
    dot.node('features', 'Feature Extraction\n(5 Conv Layers + 3 MaxPool)', fillcolor=colors['feature'])
    dot.edge('input', 'features')

    # Add fully connected layers
    dot.node('fc', 'Fully Connected Layers\n(FC-4096, FC-4096, FC-7)', fillcolor=colors['fc'])
    dot.edge('features', 'fc')

    # Add output
    dot.node('output', 'Output\n7 Classes', fillcolor=colors['output'])
    dot.edge('fc', 'output')

    return dot

def visualize_condensed_network():
    dot = create_condensed_network_visualization()
    dot.render('./Project2/image/AlexNetStruct/condensed_alexnet_architecture', format='png', cleanup=True)
    print("Condensed network visualization saved as 'condensed_alexnet_architecture.png'")

if __name__ == "__main__":
    visualize_condensed_network()