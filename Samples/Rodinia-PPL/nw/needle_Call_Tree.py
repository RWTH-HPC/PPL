from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('Shsu9axpg2lDkJm4sO5P', 'Main')
dot.node('hBpfnchCtxmGvHPtDfAq', 'Call: init_List', style="filled", fillcolor="green")
dot.node('EjePrL29v530G316xHqu', 'Call: init_List', style="filled", fillcolor="green")
dot.node('4tkD4gUxZ1x7CXYuudGq', 'Call: rand', style="filled", fillcolor="green")
dot.node('a0QukBeD4sDn2Co84nSk', 'Call: rand', style="filled", fillcolor="green")
dot.node('gHXU3FMSQSebev5J9PXN', 'Call: init_List', style="filled", fillcolor="green")
dot.node('hok18F9iRNvPIG9gAbVI', 'Call: init_List', style="filled", fillcolor="green")
dot.node('pIpsoao9O0SOHyfTRQVB', 'Map: iteration', style="filled", fillcolor="red")
dot.node('NMSnYZbMM9ka0WKPxEUJ', 'Call: maximum_arr', style="filled", fillcolor="green")
dot.node('1Gxrs9a6D3ZCzMSnraan', 'Call: maximum', style="filled", fillcolor="green")
dot.edge('Shsu9axpg2lDkJm4sO5P', 'hBpfnchCtxmGvHPtDfAq')
dot.edge('Shsu9axpg2lDkJm4sO5P', 'EjePrL29v530G316xHqu')
dot.edge('Shsu9axpg2lDkJm4sO5P', '4tkD4gUxZ1x7CXYuudGq')
dot.edge('Shsu9axpg2lDkJm4sO5P', 'a0QukBeD4sDn2Co84nSk')
dot.edge('Shsu9axpg2lDkJm4sO5P', 'gHXU3FMSQSebev5J9PXN')
dot.edge('Shsu9axpg2lDkJm4sO5P', 'hok18F9iRNvPIG9gAbVI')
dot.edge('Shsu9axpg2lDkJm4sO5P', 'pIpsoao9O0SOHyfTRQVB')
dot.edge('pIpsoao9O0SOHyfTRQVB', 'NMSnYZbMM9ka0WKPxEUJ')
dot.edge('Shsu9axpg2lDkJm4sO5P', '1Gxrs9a6D3ZCzMSnraan')



print(dot.source)
dot.render('needle_Call_Tree.gv', view=True)