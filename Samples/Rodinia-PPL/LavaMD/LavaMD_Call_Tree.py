from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('0rkiLByCEx5q4Px2m4P5', 'Main')
dot.node('V4SrjiqSY2jtlB7WFt0t', 'Call: init_List', style="filled", fillcolor="green")
dot.node('XiSMb2I9gtVigySH3LH5', 'Call: init_List', style="filled", fillcolor="green")
dot.node('cvNJ1GRzgQdrRWMsi5a3', 'Call: init_List', style="filled", fillcolor="green")
dot.node('HSpT6FAi0MsOSOsIfNbE', 'Stencil: kernel', style="filled", fillcolor="red")
dot.node('5goJvYgGe6uU57RXmNj2', 'Call: init_List', style="filled", fillcolor="green")
dot.node('PWjdDOsEBxs1Wjxcc3sV', 'Call: init_List', style="filled", fillcolor="green")
dot.node('EvSczlnvIMgQ7IjKRXxR', 'Map: particle_iteration', style="filled", fillcolor="red")
dot.node('HaxCKZxZ7PPhVXdSNhpj', 'Call: force_combination', style="filled", fillcolor="green")
dot.node('mrImu8PhKepUg1riAmG4', 'Call: init_List', style="filled", fillcolor="green")
dot.node('xO8YgHJt9NNXjrDUE1CY', 'Call: exp', style="filled", fillcolor="green")
dot.node('lJr1rbNaWIlhZHIZ3ZEQ', 'Call: powi', style="filled", fillcolor="green")
dot.node('6HT7IlrrTi9Ze7tMarYU', 'Call: powi', style="filled", fillcolor="green")
dot.node('kvRAWOTbrWaKLKjby0c0', 'Call: powi', style="filled", fillcolor="green")
dot.node('ODhE7FWW0qVzY3HrCPGu', 'Call: powi', style="filled", fillcolor="green")
dot.edge('0rkiLByCEx5q4Px2m4P5', 'V4SrjiqSY2jtlB7WFt0t')
dot.edge('0rkiLByCEx5q4Px2m4P5', 'XiSMb2I9gtVigySH3LH5')
dot.edge('0rkiLByCEx5q4Px2m4P5', 'cvNJ1GRzgQdrRWMsi5a3')
dot.edge('0rkiLByCEx5q4Px2m4P5', 'HSpT6FAi0MsOSOsIfNbE')
dot.edge('HSpT6FAi0MsOSOsIfNbE', '5goJvYgGe6uU57RXmNj2')
dot.edge('HSpT6FAi0MsOSOsIfNbE', 'PWjdDOsEBxs1Wjxcc3sV')
dot.edge('HSpT6FAi0MsOSOsIfNbE', 'EvSczlnvIMgQ7IjKRXxR')
dot.edge('EvSczlnvIMgQ7IjKRXxR', 'HaxCKZxZ7PPhVXdSNhpj')
dot.edge('HaxCKZxZ7PPhVXdSNhpj', 'mrImu8PhKepUg1riAmG4')
dot.edge('HaxCKZxZ7PPhVXdSNhpj', 'xO8YgHJt9NNXjrDUE1CY')
dot.edge('xO8YgHJt9NNXjrDUE1CY', 'lJr1rbNaWIlhZHIZ3ZEQ')
dot.edge('xO8YgHJt9NNXjrDUE1CY', '6HT7IlrrTi9Ze7tMarYU')
dot.edge('xO8YgHJt9NNXjrDUE1CY', 'kvRAWOTbrWaKLKjby0c0')
dot.edge('xO8YgHJt9NNXjrDUE1CY', 'ODhE7FWW0qVzY3HrCPGu')



print(dot.source)
dot.render('LavaMD_Call_Tree.gv', view=True)