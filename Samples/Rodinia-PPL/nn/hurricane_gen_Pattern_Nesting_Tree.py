from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('ijQTszbE54Ttrloji6HF', 'Main')
dot.node('I9gnHx0IKRgnVztch2kU', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('ijQTszbE54Ttrloji6HF', 'I9gnHx0IKRgnVztch2kU')



print(dot.source)
dot.render('hurricane_gen_Pattern_Nesting_Tree.gv', view=True)