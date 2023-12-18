from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('tM8QW8Y2jiQcLiCfN0OX', 'Main')
dot.node('u0ZMmWdLOBWdlDYkqUGd', 'Call: init_List', style="filled", fillcolor="green")
dot.node('0pY3r3mGzecR7EEzYNPo', 'Call: init_List', style="filled", fillcolor="green")
dot.node('ELAhE5cRI3SkaltXQkjC', 'Call: init_List', style="filled", fillcolor="green")
dot.node('4zdZxZGC2V4V1PPB1uDP', 'Call: compute_tran_temp', style="filled", fillcolor="green")
dot.node('UcCZikUocNF5p43brSlE', 'Call: init_List', style="filled", fillcolor="green")
dot.node('rNdEPrvG5CLlFjD4Ajb0', 'Call: init_List', style="filled", fillcolor="green")
dot.node('BYBnmStXf7c32WQhlK3Y', 'Stencil: add_padding_1', style="filled", fillcolor="red")
dot.node('UjDNyzIancvHhULVWORi', 'Stencil: side_north', style="filled", fillcolor="red")
dot.node('GvDzcgSj6OcYHnYMVtii', 'Stencil: side_south', style="filled", fillcolor="red")
dot.node('9kC9HKVxjplFIdGXUHTB', 'Stencil: side_east', style="filled", fillcolor="red")
dot.node('FCMvRKxcTstOl87AoNX2', 'Stencil: side_west', style="filled", fillcolor="red")
dot.node('UkqjEeYcIWJ9W4My9OHO', 'Stencil: side_top', style="filled", fillcolor="red")
dot.node('Bj4CO8eEtH5TGEEQAd6L', 'Stencil: side_bottom', style="filled", fillcolor="red")
dot.node('kBdkAtJPb6nIHi0zf7HH', 'Stencil: single_iteration', style="filled", fillcolor="red")
dot.node('51jhQOyTVx44cui7H4fe', 'Stencil: copy', style="filled", fillcolor="red")
dot.edge('tM8QW8Y2jiQcLiCfN0OX', 'u0ZMmWdLOBWdlDYkqUGd')
dot.edge('tM8QW8Y2jiQcLiCfN0OX', '0pY3r3mGzecR7EEzYNPo')
dot.edge('tM8QW8Y2jiQcLiCfN0OX', 'ELAhE5cRI3SkaltXQkjC')
dot.edge('tM8QW8Y2jiQcLiCfN0OX', '4zdZxZGC2V4V1PPB1uDP')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'UcCZikUocNF5p43brSlE')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'rNdEPrvG5CLlFjD4Ajb0')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'BYBnmStXf7c32WQhlK3Y')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'UjDNyzIancvHhULVWORi')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'GvDzcgSj6OcYHnYMVtii')
dot.edge('4zdZxZGC2V4V1PPB1uDP', '9kC9HKVxjplFIdGXUHTB')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'FCMvRKxcTstOl87AoNX2')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'UkqjEeYcIWJ9W4My9OHO')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'Bj4CO8eEtH5TGEEQAd6L')
dot.edge('4zdZxZGC2V4V1PPB1uDP', 'kBdkAtJPb6nIHi0zf7HH')
dot.edge('4zdZxZGC2V4V1PPB1uDP', '51jhQOyTVx44cui7H4fe')



print(dot.source)
dot.render('hotspot3D_Call_Tree.gv', view=True)