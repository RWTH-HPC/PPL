from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('85lrRpvIxvWxItVCZuPB', 'Main')
dot.node('2SYrhcT3rCsNODmxPpT9', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('HXKVB1OFjmXSqfSGF5gE', 'For Loop', style="filled", fillcolor="cyan")
dot.node('YkOxxMI6zYBer9ctY4UY', 'Map: deriv_N', style="filled", fillcolor="orangered")
dot.node('xzNYUXtpcYqgpJKnUydN', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('C0kiCPk8AaSzBWMSZPpI', 'Map: deriv_row', style="filled", fillcolor="orangered")
dot.node('UNISVelVlJ3McNytYnJ6', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('NQlQvFuAksrVX1qZT0Br', 'Map: deriv_S', style="filled", fillcolor="orangered")
dot.node('AvsagndiO8nTMzRKWSxs', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('0HctrVLB6Pf8sfBBO5IA', 'Map: deriv_row', style="filled", fillcolor="orangered")
dot.node('MGCkSQG9bGn7PljRjZHu', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('kAYLZAH3prZvTnh59JcE', 'Stencil: deriv_W', style="filled", fillcolor="orangered")
dot.node('ghVa4aJOBNAQZAJlzFyT', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('2d610sibmlogktNzq1gP', 'Stencil: deriv_E', style="filled", fillcolor="orangered")
dot.node('MSDN8xqlvO8g9NRohVmY', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('1UzSkhf2brqWG5IFOqNQ', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('ThxDVLK762UgJMnqWXdI', 'Stencil: diffuse', style="filled", fillcolor="orangered")
dot.node('BivFh4B5mTu6sk4ATJa0', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('80iGQZ7KB6zbDKTbYYdC', 'Stencil: update', style="filled", fillcolor="orangered")
dot.node('oKyq2yLXFtRACHoklObl', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('28CFRuBY2tfozGVbVlZa', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('85lrRpvIxvWxItVCZuPB', '2SYrhcT3rCsNODmxPpT9')
dot.edge('85lrRpvIxvWxItVCZuPB', 'HXKVB1OFjmXSqfSGF5gE')
dot.edge('HXKVB1OFjmXSqfSGF5gE', 'YkOxxMI6zYBer9ctY4UY')
dot.edge('YkOxxMI6zYBer9ctY4UY', 'xzNYUXtpcYqgpJKnUydN')
dot.edge('YkOxxMI6zYBer9ctY4UY', 'C0kiCPk8AaSzBWMSZPpI')
dot.edge('C0kiCPk8AaSzBWMSZPpI', 'UNISVelVlJ3McNytYnJ6')
dot.edge('HXKVB1OFjmXSqfSGF5gE', 'NQlQvFuAksrVX1qZT0Br')
dot.edge('NQlQvFuAksrVX1qZT0Br', 'AvsagndiO8nTMzRKWSxs')
dot.edge('NQlQvFuAksrVX1qZT0Br', '0HctrVLB6Pf8sfBBO5IA')
dot.edge('0HctrVLB6Pf8sfBBO5IA', 'MGCkSQG9bGn7PljRjZHu')
dot.edge('HXKVB1OFjmXSqfSGF5gE', 'kAYLZAH3prZvTnh59JcE')
dot.edge('kAYLZAH3prZvTnh59JcE', 'ghVa4aJOBNAQZAJlzFyT')
dot.edge('HXKVB1OFjmXSqfSGF5gE', '2d610sibmlogktNzq1gP')
dot.edge('2d610sibmlogktNzq1gP', 'MSDN8xqlvO8g9NRohVmY')
dot.edge('HXKVB1OFjmXSqfSGF5gE', '1UzSkhf2brqWG5IFOqNQ')
dot.edge('HXKVB1OFjmXSqfSGF5gE', 'ThxDVLK762UgJMnqWXdI')
dot.edge('ThxDVLK762UgJMnqWXdI', 'BivFh4B5mTu6sk4ATJa0')
dot.edge('HXKVB1OFjmXSqfSGF5gE', '80iGQZ7KB6zbDKTbYYdC')
dot.edge('80iGQZ7KB6zbDKTbYYdC', 'oKyq2yLXFtRACHoklObl')
dot.edge('85lrRpvIxvWxItVCZuPB', '28CFRuBY2tfozGVbVlZa')



print(dot.source)
dot.render('srad_Pattern_Nesting_Tree.gv', view=True)