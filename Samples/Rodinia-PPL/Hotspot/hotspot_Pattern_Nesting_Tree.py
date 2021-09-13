from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('q2r1DCZfmKzaZFdPrZRV', 'Main')
dot.node('L3Dp1Ld5kB0Pb2WDYKuM', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('g0WYktQzmmWVPBkBA1vy', 'Stencil: copy', style="filled", fillcolor="orangered")
dot.node('2TKkEDZmvQOseR55Yz1W', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('a6XsufMq0hbWmzbfkVDA', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('Cq0UcliWgaDJzaLbbeV5', 'For Loop', style="filled", fillcolor="cyan")
dot.node('3ZvyIJAGa8xRxnUbv4ST', 'Stencil: edge1', style="filled", fillcolor="orangered")
dot.node('F8LLbhVzkTt82oSp5um0', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('sqnMBZ5e9iuAOkL3EHoN', 'Stencil: edge2', style="filled", fillcolor="orangered")
dot.node('tKU7W1rVoMn8iFyTv71H', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('3F0qXAu43feOPSNtZb0g', 'Stencil: edge3', style="filled", fillcolor="orangered")
dot.node('U3aZWL2URKAQGLpuCEUg', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('AMIHSAtGl35x3EI4I7J9', 'Stencil: edge4', style="filled", fillcolor="orangered")
dot.node('1IJWSe8onuJZsEBfldRB', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('8DmJp7gv2fFmaCyB26sD', 'Stencil: single_iteration', style="filled", fillcolor="orangered")
dot.node('6Vxr8yGRrfq5zfcfYUzc', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('HPQAC9lEnakaji24AEEv', 'Stencil: copy', style="filled", fillcolor="orangered")
dot.node('gGIUv9AyEkcfmO9noxc5', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('BRiaLDC38S6htDKRbrd0', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('q2r1DCZfmKzaZFdPrZRV', 'L3Dp1Ld5kB0Pb2WDYKuM')
dot.edge('q2r1DCZfmKzaZFdPrZRV', 'g0WYktQzmmWVPBkBA1vy')
dot.edge('g0WYktQzmmWVPBkBA1vy', '2TKkEDZmvQOseR55Yz1W')
dot.edge('q2r1DCZfmKzaZFdPrZRV', 'a6XsufMq0hbWmzbfkVDA')
dot.edge('q2r1DCZfmKzaZFdPrZRV', 'Cq0UcliWgaDJzaLbbeV5')
dot.edge('Cq0UcliWgaDJzaLbbeV5', '3ZvyIJAGa8xRxnUbv4ST')
dot.edge('3ZvyIJAGa8xRxnUbv4ST', 'F8LLbhVzkTt82oSp5um0')
dot.edge('Cq0UcliWgaDJzaLbbeV5', 'sqnMBZ5e9iuAOkL3EHoN')
dot.edge('sqnMBZ5e9iuAOkL3EHoN', 'tKU7W1rVoMn8iFyTv71H')
dot.edge('Cq0UcliWgaDJzaLbbeV5', '3F0qXAu43feOPSNtZb0g')
dot.edge('3F0qXAu43feOPSNtZb0g', 'U3aZWL2URKAQGLpuCEUg')
dot.edge('Cq0UcliWgaDJzaLbbeV5', 'AMIHSAtGl35x3EI4I7J9')
dot.edge('AMIHSAtGl35x3EI4I7J9', '1IJWSe8onuJZsEBfldRB')
dot.edge('Cq0UcliWgaDJzaLbbeV5', '8DmJp7gv2fFmaCyB26sD')
dot.edge('8DmJp7gv2fFmaCyB26sD', '6Vxr8yGRrfq5zfcfYUzc')
dot.edge('Cq0UcliWgaDJzaLbbeV5', 'HPQAC9lEnakaji24AEEv')
dot.edge('HPQAC9lEnakaji24AEEv', 'gGIUv9AyEkcfmO9noxc5')
dot.edge('q2r1DCZfmKzaZFdPrZRV', 'BRiaLDC38S6htDKRbrd0')



print(dot.source)
dot.render('hotspot_Pattern_Nesting_Tree.gv', view=True)