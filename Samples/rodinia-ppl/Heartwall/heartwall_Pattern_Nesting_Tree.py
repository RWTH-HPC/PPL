from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('PdJBZaDUrx968q6OIEqJ', 'Main')
dot.node('DYdmP3MibYk8q4XDWUOL', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('gOrs4Kz4kePBmBhFiRNu', 'For Loop', style="filled", fillcolor="cyan")
dot.node('BjGhmWICW4gnVVcRZLhB', 'Stencil: multi_copy', style="filled", fillcolor="orangered")
dot.node('tJocTYrXX2NtTNug7CQy', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('eH7QOpLxXckkl6XRmeBQ', 'Map: kernel', style="filled", fillcolor="orangered")
dot.node('2Vxjbba8Wmk9vaFD8Ut7', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('X2xT7NO6760miGfhLA3J', 'Map: copy', style="filled", fillcolor="orangered")
dot.node('mv3tniMancJWv5SsFtLu', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('kWRxIBHfMW1VZm2SfnAv', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('cKZjpHk7g7kHO5X6jKsi', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('PdJBZaDUrx968q6OIEqJ', 'DYdmP3MibYk8q4XDWUOL')
dot.edge('PdJBZaDUrx968q6OIEqJ', 'gOrs4Kz4kePBmBhFiRNu')
dot.edge('gOrs4Kz4kePBmBhFiRNu', 'BjGhmWICW4gnVVcRZLhB')
dot.edge('BjGhmWICW4gnVVcRZLhB', 'tJocTYrXX2NtTNug7CQy')
dot.edge('gOrs4Kz4kePBmBhFiRNu', 'eH7QOpLxXckkl6XRmeBQ')
dot.edge('eH7QOpLxXckkl6XRmeBQ', '2Vxjbba8Wmk9vaFD8Ut7')
dot.edge('eH7QOpLxXckkl6XRmeBQ', 'X2xT7NO6760miGfhLA3J')
dot.edge('X2xT7NO6760miGfhLA3J', 'mv3tniMancJWv5SsFtLu')
dot.edge('eH7QOpLxXckkl6XRmeBQ', 'kWRxIBHfMW1VZm2SfnAv')
dot.edge('PdJBZaDUrx968q6OIEqJ', 'cKZjpHk7g7kHO5X6jKsi')



print(dot.source)
dot.render('heartwall_Pattern_Nesting_Tree.gv', view=True)