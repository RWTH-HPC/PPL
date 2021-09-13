from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('t4Xf5NS6KFmbplIYS6ME', 'Main')
dot.node('XExAScSNTjzQaXe3zCXz', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('gIcrdXhi5WAlPJX28Xbt', 'Map: kernel', style="filled", fillcolor="orangered")
dot.node('roOGNvuy5B75NkD3QIUh', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('KPOBQTRNJJL1mUFzSeMJ', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('t4Xf5NS6KFmbplIYS6ME', 'XExAScSNTjzQaXe3zCXz')
dot.edge('t4Xf5NS6KFmbplIYS6ME', 'gIcrdXhi5WAlPJX28Xbt')
dot.edge('gIcrdXhi5WAlPJX28Xbt', 'roOGNvuy5B75NkD3QIUh')
dot.edge('t4Xf5NS6KFmbplIYS6ME', 'KPOBQTRNJJL1mUFzSeMJ')



print(dot.source)
dot.render('myocyte_Pattern_Nesting_Tree.gv', view=True)