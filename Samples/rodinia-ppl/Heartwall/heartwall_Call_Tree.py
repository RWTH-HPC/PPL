from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('qoltQkNHn6ntaMHhbtGq', 'Main')
dot.node('ZAGEiWlQz6xgpuCRz9HZ', 'Call: init_List', style="filled", fillcolor="green")
dot.node('sD51Dfx9Rmy7EGUUXOsT', 'Call: init_List', style="filled", fillcolor="green")
dot.node('bA2nGczl5E2YkNKYbWPV', 'Call: init_List', style="filled", fillcolor="green")
dot.node('CpshDdlMx0mucFdzNsvZ', 'Stencil: multi_copy', style="filled", fillcolor="red")
dot.node('QIdOKdP4ZFlASNqivecf', 'Map: kernel', style="filled", fillcolor="red")
dot.node('M6Y5FV5EJcdyJfXUlKmI', 'Call: kernel_func', style="filled", fillcolor="green")
dot.node('T072kt7Jf7u1Jn6yJUx5', 'Call: init_List', style="filled", fillcolor="green")
dot.node('usq4Qxmlr3C2k9F1Amz3', 'Map: copy', style="filled", fillcolor="red")
dot.node('P7eXpB4pv3uX4CIIoJfp', 'Call: init_List', style="filled", fillcolor="green")
dot.node('UsKma7gfdK2Ap1TK5VMo', 'Call: init_List', style="filled", fillcolor="green")
dot.node('gZybt0lLQJFvm55M36FP', 'Call: init_List', style="filled", fillcolor="green")
dot.node('4VeGD84WnfLaC6zceFLZ', 'Call: init_List', style="filled", fillcolor="green")
dot.node('jhI8GclZL3USIbCa2rq9', 'Call: init_List', style="filled", fillcolor="green")
dot.node('2FOjcdLCTnTcigDmCPhv', 'Call: init_List', style="filled", fillcolor="green")
dot.node('2IqucX7228PyB4ACBIUf', 'Call: init_List', style="filled", fillcolor="green")
dot.node('2563yi4lMBs2vDsT7b0m', 'Call: init_List', style="filled", fillcolor="green")
dot.node('KdOuymBuiHQyke86cyDd', 'Call: init_List', style="filled", fillcolor="green")
dot.node('RUoYWBDLP72xsxgBDvsS', 'Call: init_List', style="filled", fillcolor="green")
dot.node('bycVUKm8gQH63cVpIjP1', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('W5TC4XmUo6k44KqG4YnK', 'Call: sqrt', style="filled", fillcolor="green")
dot.edge('qoltQkNHn6ntaMHhbtGq', 'ZAGEiWlQz6xgpuCRz9HZ')
dot.edge('qoltQkNHn6ntaMHhbtGq', 'sD51Dfx9Rmy7EGUUXOsT')
dot.edge('qoltQkNHn6ntaMHhbtGq', 'bA2nGczl5E2YkNKYbWPV')
dot.edge('qoltQkNHn6ntaMHhbtGq', 'CpshDdlMx0mucFdzNsvZ')
dot.edge('qoltQkNHn6ntaMHhbtGq', 'QIdOKdP4ZFlASNqivecf')
dot.edge('QIdOKdP4ZFlASNqivecf', 'M6Y5FV5EJcdyJfXUlKmI')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'T072kt7Jf7u1Jn6yJUx5')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'usq4Qxmlr3C2k9F1Amz3')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'P7eXpB4pv3uX4CIIoJfp')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'UsKma7gfdK2Ap1TK5VMo')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'gZybt0lLQJFvm55M36FP')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', '4VeGD84WnfLaC6zceFLZ')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'jhI8GclZL3USIbCa2rq9')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', '2FOjcdLCTnTcigDmCPhv')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', '2IqucX7228PyB4ACBIUf')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', '2563yi4lMBs2vDsT7b0m')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'KdOuymBuiHQyke86cyDd')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'RUoYWBDLP72xsxgBDvsS')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'bycVUKm8gQH63cVpIjP1')
dot.edge('M6Y5FV5EJcdyJfXUlKmI', 'W5TC4XmUo6k44KqG4YnK')



print(dot.source)
dot.render('heartwall_Call_Tree.gv', view=True)