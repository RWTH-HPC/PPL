from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('yNHs5shAMHcb6ZP8sK8L', 'Main')
dot.node('fC1FQeejG3FaFu1YfMC5', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('6oTrZp1SY2XyoNDx5BKd', 'Stencil: kernel', style="filled", fillcolor="orangered")
dot.node('PYyZgTTbupSe6YvgfAEK', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('B6RXtomA9ExPJlP5RRkH', 'Map: particle_iteration', style="filled", fillcolor="orangered")
dot.node('Qh7aDIiUX05yR2XYfqnv', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('yNHs5shAMHcb6ZP8sK8L', 'fC1FQeejG3FaFu1YfMC5')
dot.edge('yNHs5shAMHcb6ZP8sK8L', '6oTrZp1SY2XyoNDx5BKd')
dot.edge('6oTrZp1SY2XyoNDx5BKd', 'PYyZgTTbupSe6YvgfAEK')
dot.edge('6oTrZp1SY2XyoNDx5BKd', 'B6RXtomA9ExPJlP5RRkH')
dot.edge('B6RXtomA9ExPJlP5RRkH', 'Qh7aDIiUX05yR2XYfqnv')



print(dot.source)
dot.render('LavaMD_Pattern_Nesting_Tree.gv', view=True)