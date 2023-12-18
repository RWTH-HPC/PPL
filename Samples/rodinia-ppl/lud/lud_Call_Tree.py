from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('iYilthD9kY9hCUxSAlDC', 'Main')
dot.node('WGURaHMBTghrcHEDYWJp', 'Call: init_List', style="filled", fillcolor="green")
dot.node('TQ8jgZ3bjGEVER6QYZUK', 'Call: init_List', style="filled", fillcolor="green")
dot.node('e94R9ZXk5hjmOzZJ07ny', 'Map: kernel', style="filled", fillcolor="red")
dot.node('sRKlV15vm5AVCaSFK4qD', 'Call: handle_row', style="filled", fillcolor="green")
dot.node('cbLn36bA21DEB3p1R5LN', 'Call: init_List', style="filled", fillcolor="green")
dot.edge('iYilthD9kY9hCUxSAlDC', 'WGURaHMBTghrcHEDYWJp')
dot.edge('iYilthD9kY9hCUxSAlDC', 'TQ8jgZ3bjGEVER6QYZUK')
dot.edge('iYilthD9kY9hCUxSAlDC', 'e94R9ZXk5hjmOzZJ07ny')
dot.edge('e94R9ZXk5hjmOzZJ07ny', 'sRKlV15vm5AVCaSFK4qD')
dot.edge('sRKlV15vm5AVCaSFK4qD', 'cbLn36bA21DEB3p1R5LN')



print(dot.source)
dot.render('lud_Call_Tree.gv', view=True)