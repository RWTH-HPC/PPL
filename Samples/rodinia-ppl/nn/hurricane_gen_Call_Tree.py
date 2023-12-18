from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('DVMNAleKVeFDKgqXn898', 'Main')
dot.node('AWHystuMmDdTkVllaejQ', 'Call: rand', style="filled", fillcolor="green")
dot.node('jyLLBfTGBX3K461bJegI', 'Call: rand', style="filled", fillcolor="green")
dot.node('glSrJbzqA3qWGjvWNfPJ', 'Call: rand', style="filled", fillcolor="green")
dot.node('kRE7nH1Y3LJN4YpbX5T6', 'Call: rand', style="filled", fillcolor="green")
dot.node('DMyHlOnRVVJgByEPRvD7', 'Call: rand', style="filled", fillcolor="green")
dot.node('dDUfF6VCB0xZrRA7LBUj', 'Call: rand', style="filled", fillcolor="green")
dot.node('pluidRv0NTiyzerjr8Q6', 'Call: rand', style="filled", fillcolor="green")
dot.node('ukx0W3xIpYUA8nrgQVHi', 'Call: rand', style="filled", fillcolor="green")
dot.node('2E9Q1jUE7MhyVu1hkcMC', 'Call: rand', style="filled", fillcolor="green")
dot.node('5RMQ1aRKdrUTN3qOyRaU', 'Call: rand', style="filled", fillcolor="green")
dot.node('oyvhZBOoiFB6eD619UpA', 'Call: rand', style="filled", fillcolor="green")
dot.node('MfVQEidUAfeLfOHgjU4M', 'Call: rand', style="filled", fillcolor="green")
dot.edge('DVMNAleKVeFDKgqXn898', 'AWHystuMmDdTkVllaejQ')
dot.edge('DVMNAleKVeFDKgqXn898', 'jyLLBfTGBX3K461bJegI')
dot.edge('DVMNAleKVeFDKgqXn898', 'glSrJbzqA3qWGjvWNfPJ')
dot.edge('DVMNAleKVeFDKgqXn898', 'kRE7nH1Y3LJN4YpbX5T6')
dot.edge('DVMNAleKVeFDKgqXn898', 'DMyHlOnRVVJgByEPRvD7')
dot.edge('DVMNAleKVeFDKgqXn898', 'dDUfF6VCB0xZrRA7LBUj')
dot.edge('DVMNAleKVeFDKgqXn898', 'pluidRv0NTiyzerjr8Q6')
dot.edge('DVMNAleKVeFDKgqXn898', 'ukx0W3xIpYUA8nrgQVHi')
dot.edge('DVMNAleKVeFDKgqXn898', '2E9Q1jUE7MhyVu1hkcMC')
dot.edge('DVMNAleKVeFDKgqXn898', '5RMQ1aRKdrUTN3qOyRaU')
dot.edge('DVMNAleKVeFDKgqXn898', 'oyvhZBOoiFB6eD619UpA')
dot.edge('DVMNAleKVeFDKgqXn898', 'MfVQEidUAfeLfOHgjU4M')



print(dot.source)
dot.render('hurricane_gen_Call_Tree.gv', view=True)