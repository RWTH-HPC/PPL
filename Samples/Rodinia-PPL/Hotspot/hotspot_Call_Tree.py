from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('dNZV2xWGoeRSgAp0HaKV', 'Main')
dot.node('JE2THdY6nVFbSbikzrFs', 'Call: init_List', style="filled", fillcolor="green")
dot.node('rhh0PJxT5EaGChc17Fhh', 'Call: init_List', style="filled", fillcolor="green")
dot.node('iNPB7GonxiavRHD77ACj', 'Call: init_List', style="filled", fillcolor="green")
dot.node('541d9pH3GDK2CEBbZpkF', 'Call: compute_tran_temp', style="filled", fillcolor="green")
dot.node('1bW6BePfP8PnErtml23B', 'Call: init_List', style="filled", fillcolor="green")
dot.node('tWk5TMYzq34KTl8pMVHY', 'Call: init_List', style="filled", fillcolor="green")
dot.node('B85CjC4kkQIS2LuoBpXO', 'Stencil: copy', style="filled", fillcolor="red")
dot.node('QYzytQ9yjT3rUfLB8HW8', 'Stencil: edge1', style="filled", fillcolor="red")
dot.node('6VHZdGjiYcsk4XYVujqY', 'Stencil: edge2', style="filled", fillcolor="red")
dot.node('Zr8s3pv0yOfoDLHsdU3I', 'Stencil: edge3', style="filled", fillcolor="red")
dot.node('aDeRAYnue9DbzBbWlAuM', 'Stencil: edge4', style="filled", fillcolor="red")
dot.node('P3Rhcu8V2QSHzPaQOhrM', 'Stencil: single_iteration', style="filled", fillcolor="red")
dot.node('4RSsoAjvJELtyGfIVpK8', 'Stencil: copy', style="filled", fillcolor="red")
dot.edge('dNZV2xWGoeRSgAp0HaKV', 'JE2THdY6nVFbSbikzrFs')
dot.edge('dNZV2xWGoeRSgAp0HaKV', 'rhh0PJxT5EaGChc17Fhh')
dot.edge('dNZV2xWGoeRSgAp0HaKV', 'iNPB7GonxiavRHD77ACj')
dot.edge('dNZV2xWGoeRSgAp0HaKV', '541d9pH3GDK2CEBbZpkF')
dot.edge('541d9pH3GDK2CEBbZpkF', '1bW6BePfP8PnErtml23B')
dot.edge('541d9pH3GDK2CEBbZpkF', 'tWk5TMYzq34KTl8pMVHY')
dot.edge('541d9pH3GDK2CEBbZpkF', 'B85CjC4kkQIS2LuoBpXO')
dot.edge('541d9pH3GDK2CEBbZpkF', 'QYzytQ9yjT3rUfLB8HW8')
dot.edge('541d9pH3GDK2CEBbZpkF', '6VHZdGjiYcsk4XYVujqY')
dot.edge('541d9pH3GDK2CEBbZpkF', 'Zr8s3pv0yOfoDLHsdU3I')
dot.edge('541d9pH3GDK2CEBbZpkF', 'aDeRAYnue9DbzBbWlAuM')
dot.edge('541d9pH3GDK2CEBbZpkF', 'P3Rhcu8V2QSHzPaQOhrM')
dot.edge('541d9pH3GDK2CEBbZpkF', '4RSsoAjvJELtyGfIVpK8')



print(dot.source)
dot.render('hotspot_Call_Tree.gv', view=True)