from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('Zte1Pvv7dDiP2mmPGQlN', 'Main')
dot.node('aNVQONGzsj1gb4VgQHZ3', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('vzEGFu0YcPX2jXUs4UbG', 'For Loop', style="filled", fillcolor="cyan")
dot.node('PLe2S28eZlAJeUj5Ftst', 'Map: kernel', style="filled", fillcolor="orangered")
dot.node('IAmaoko1ZmBsfoHa1ZhL', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('OWoB4X8pvMiY33ZhOADv', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('Zte1Pvv7dDiP2mmPGQlN', 'aNVQONGzsj1gb4VgQHZ3')
dot.edge('Zte1Pvv7dDiP2mmPGQlN', 'vzEGFu0YcPX2jXUs4UbG')
dot.edge('vzEGFu0YcPX2jXUs4UbG', 'PLe2S28eZlAJeUj5Ftst')
dot.edge('PLe2S28eZlAJeUj5Ftst', 'IAmaoko1ZmBsfoHa1ZhL')
dot.edge('vzEGFu0YcPX2jXUs4UbG', 'OWoB4X8pvMiY33ZhOADv')



print(dot.source)
dot.render('pathfinder_Pattern_Nesting_Tree.gv', view=True)