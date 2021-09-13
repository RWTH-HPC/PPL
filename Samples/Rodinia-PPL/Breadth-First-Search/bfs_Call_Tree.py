from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('SrcPCXhr8MxAJu6mrevQ', 'Main')
dot.node('9d2YJnqTbzyupFLyBtKg', 'Call: init_List', style="filled", fillcolor="green")
dot.node('lJ9d10dYhrlE4osGBRMe', 'Call: init_List', style="filled", fillcolor="green")
dot.node('KthxqT8oZPbKq3qp6eui', 'Call: init_List', style="filled", fillcolor="green")
dot.node('22REpNYTFTH6ITKprBkS', 'Call: init_List', style="filled", fillcolor="green")
dot.node('Yo6okQGHA6CeDyNM5reR', 'Call: init_List', style="filled", fillcolor="green")
dot.node('neAG3IyobqIQOYuTcP8Z', 'Call: init_List', style="filled", fillcolor="green")
dot.node('KxUdvtVAVFNylJm4RFtb', 'Map: update_visit', style="filled", fillcolor="red")
dot.node('nhgBIYIKjUhgutfDIpnn', 'Reduction: isFinished', style="filled", fillcolor="red")
dot.node('PmYxk2W1naktPv8321mn', 'Map: reset', style="filled", fillcolor="red")
dot.edge('SrcPCXhr8MxAJu6mrevQ', '9d2YJnqTbzyupFLyBtKg')
dot.edge('SrcPCXhr8MxAJu6mrevQ', 'lJ9d10dYhrlE4osGBRMe')
dot.edge('SrcPCXhr8MxAJu6mrevQ', 'KthxqT8oZPbKq3qp6eui')
dot.edge('SrcPCXhr8MxAJu6mrevQ', '22REpNYTFTH6ITKprBkS')
dot.edge('SrcPCXhr8MxAJu6mrevQ', 'Yo6okQGHA6CeDyNM5reR')
dot.edge('SrcPCXhr8MxAJu6mrevQ', 'neAG3IyobqIQOYuTcP8Z')
dot.edge('SrcPCXhr8MxAJu6mrevQ', 'KxUdvtVAVFNylJm4RFtb')
dot.edge('SrcPCXhr8MxAJu6mrevQ', 'nhgBIYIKjUhgutfDIpnn')
dot.edge('SrcPCXhr8MxAJu6mrevQ', 'PmYxk2W1naktPv8321mn')



print(dot.source)
dot.render('bfs_Call_Tree.gv', view=True)