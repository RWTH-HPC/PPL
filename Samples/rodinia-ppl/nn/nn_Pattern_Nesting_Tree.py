from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('FZlkPZtMPAdCxrCG5PZz', 'Main')
dot.node('ipMrf8dCdMGzPrEQmiHm', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('198tCvVOlvQTuRPeUHla', 'Map: kernel', style="filled", fillcolor="orangered")
dot.node('EuVRLIpSzUSiy9qmOvEb', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('ihP0fWvDqOZasls7WHAB', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('FZlkPZtMPAdCxrCG5PZz', 'ipMrf8dCdMGzPrEQmiHm')
dot.edge('FZlkPZtMPAdCxrCG5PZz', '198tCvVOlvQTuRPeUHla')
dot.edge('198tCvVOlvQTuRPeUHla', 'EuVRLIpSzUSiy9qmOvEb')
dot.edge('FZlkPZtMPAdCxrCG5PZz', 'ihP0fWvDqOZasls7WHAB')



print(dot.source)
dot.render('nn_Pattern_Nesting_Tree.gv', view=True)