from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('HP5AaTg2OY8TZYMl6sCF', 'Main')
dot.node('8bruUfuaXaYUryojiEHO', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('4vxYdm4P7yIDyA67HmIh', 'For Loop', style="filled", fillcolor="cyan")
dot.node('j8YnBCaH3yDOhPro9tcO', 'Map: kernel', style="filled", fillcolor="orangered")
dot.node('UYSofju4qN0452HhxMjZ', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('ennzKmbDS4iodP21huxG', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('HP5AaTg2OY8TZYMl6sCF', '8bruUfuaXaYUryojiEHO')
dot.edge('HP5AaTg2OY8TZYMl6sCF', '4vxYdm4P7yIDyA67HmIh')
dot.edge('4vxYdm4P7yIDyA67HmIh', 'j8YnBCaH3yDOhPro9tcO')
dot.edge('j8YnBCaH3yDOhPro9tcO', 'UYSofju4qN0452HhxMjZ')
dot.edge('4vxYdm4P7yIDyA67HmIh', 'ennzKmbDS4iodP21huxG')



print(dot.source)
dot.render('lud_Pattern_Nesting_Tree.gv', view=True)