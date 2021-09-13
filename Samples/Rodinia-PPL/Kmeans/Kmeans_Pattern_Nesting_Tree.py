from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('tuPKLknWfLKyv0CRU4TD', 'Main')
dot.node('sOc6r1MyfycJncjDyWgj', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('XaeGQASelX6LLLkfQ5iD', 'Map: copy', style="filled", fillcolor="orangered")
dot.node('fCBKenMJQQgaHDB7G6Gp', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('lgAz5yDoK7AAXsfNpAfr', 'For Loop', style="filled", fillcolor="cyan")
dot.node('ROb2YO1d9YRKn09gekJl', 'Map: determine_cemtroids', style="filled", fillcolor="orangered")
dot.node('GAM6EbFyQZ9dKlSGXuSt', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('jjWxG0F66LaRCQNdf4XG', 'Map: update_centroids', style="filled", fillcolor="orangered")
dot.node('lhyR3P5a1O0sCx9NzMfB', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('hcoSrJOhVoTKiEE2mJud', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('tuPKLknWfLKyv0CRU4TD', 'sOc6r1MyfycJncjDyWgj')
dot.edge('tuPKLknWfLKyv0CRU4TD', 'XaeGQASelX6LLLkfQ5iD')
dot.edge('XaeGQASelX6LLLkfQ5iD', 'fCBKenMJQQgaHDB7G6Gp')
dot.edge('tuPKLknWfLKyv0CRU4TD', 'lgAz5yDoK7AAXsfNpAfr')
dot.edge('lgAz5yDoK7AAXsfNpAfr', 'ROb2YO1d9YRKn09gekJl')
dot.edge('ROb2YO1d9YRKn09gekJl', 'GAM6EbFyQZ9dKlSGXuSt')
dot.edge('lgAz5yDoK7AAXsfNpAfr', 'jjWxG0F66LaRCQNdf4XG')
dot.edge('jjWxG0F66LaRCQNdf4XG', 'lhyR3P5a1O0sCx9NzMfB')
dot.edge('tuPKLknWfLKyv0CRU4TD', 'hcoSrJOhVoTKiEE2mJud')



print(dot.source)
dot.render('Kmeans_Pattern_Nesting_Tree.gv', view=True)