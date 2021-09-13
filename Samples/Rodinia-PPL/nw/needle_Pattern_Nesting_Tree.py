from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('JPbpFW6ozzx6fuG956Q8', 'Main')
dot.node('k1yoydlq5HO8dPhPYBQP', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('EEKd1yHoWLQMpI9cDpKn', 'For Loop', style="filled", fillcolor="cyan")
dot.node('kr4YGixzCASuPbLRC4yT', 'Map: iteration', style="filled", fillcolor="orangered")
dot.node('dr0thaeofR8jrWpYD1NT', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('L4mBsVkUJv6vTzrmSUgb', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('JPbpFW6ozzx6fuG956Q8', 'k1yoydlq5HO8dPhPYBQP')
dot.edge('JPbpFW6ozzx6fuG956Q8', 'EEKd1yHoWLQMpI9cDpKn')
dot.edge('EEKd1yHoWLQMpI9cDpKn', 'kr4YGixzCASuPbLRC4yT')
dot.edge('kr4YGixzCASuPbLRC4yT', 'dr0thaeofR8jrWpYD1NT')
dot.edge('EEKd1yHoWLQMpI9cDpKn', 'L4mBsVkUJv6vTzrmSUgb')



print(dot.source)
dot.render('needle_Pattern_Nesting_Tree.gv', view=True)