from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('i89VKjd2PPb5gRcolBP1', 'Main')
dot.node('gM8INJRu7kaDiboybgBi', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('IBNnLkcNucXOkJ60rEdx', 'While Loop', style="filled", fillcolor="cyan")
dot.node('Ru4Kn5DoCvuR7RiBLcLF', 'Map: update_visit', style="filled", fillcolor="orangered")
dot.node('FBqVDqA9OOKYqepz3dNS', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('Y8pgk6gKii8k0Q73KQYV', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('a9pEtUsx4LZZHuoImAm3', 'Reduction: isFinished', style="filled", fillcolor="orangered")
dot.node('10JHlBBPE4ChzUY4y5ZA', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('lPEqvxJI1ULzV2XE8dxR', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('DLxoPhlfdqkFVF3UCgYk', 'Map: reset', style="filled", fillcolor="orangered")
dot.node('YTDVJBkCHbggCOHNOqoz', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('p2xusa3B1HlpcBUzeKPA', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('i89VKjd2PPb5gRcolBP1', 'gM8INJRu7kaDiboybgBi')
dot.edge('i89VKjd2PPb5gRcolBP1', 'IBNnLkcNucXOkJ60rEdx')
dot.edge('IBNnLkcNucXOkJ60rEdx', 'Ru4Kn5DoCvuR7RiBLcLF')
dot.edge('Ru4Kn5DoCvuR7RiBLcLF', 'FBqVDqA9OOKYqepz3dNS')
dot.edge('IBNnLkcNucXOkJ60rEdx', 'Y8pgk6gKii8k0Q73KQYV')
dot.edge('IBNnLkcNucXOkJ60rEdx', 'a9pEtUsx4LZZHuoImAm3')
dot.edge('a9pEtUsx4LZZHuoImAm3', '10JHlBBPE4ChzUY4y5ZA')
dot.edge('IBNnLkcNucXOkJ60rEdx', 'lPEqvxJI1ULzV2XE8dxR')
dot.edge('IBNnLkcNucXOkJ60rEdx', 'DLxoPhlfdqkFVF3UCgYk')
dot.edge('DLxoPhlfdqkFVF3UCgYk', 'YTDVJBkCHbggCOHNOqoz')
dot.edge('i89VKjd2PPb5gRcolBP1', 'p2xusa3B1HlpcBUzeKPA')



print(dot.source)
dot.render('bfs_Pattern_Nesting_Tree.gv', view=True)