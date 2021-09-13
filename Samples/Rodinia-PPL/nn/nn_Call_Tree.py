from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('ktZZFrrMGHOnrk7m0ACZ', 'Main')
dot.node('ALUCx51bmHH1HgAIf7o1', 'Call: init_List', style="filled", fillcolor="green")
dot.node('kme7ittgG00LuD5h7qbO', 'Call: init_List', style="filled", fillcolor="green")
dot.node('xNEIrJyFnm3mYPdmSuWI', 'Call: init_List', style="filled", fillcolor="green")
dot.node('XkAiVGFt3I9y6ZIEKibL', 'Call: init_List', style="filled", fillcolor="green")
dot.node('0tb0X4Gjm86PusO2J2SB', 'Call: init_List', style="filled", fillcolor="green")
dot.node('GhiRA0fJBKNdPFkHN2D1', 'Call: init_List', style="filled", fillcolor="green")
dot.node('9IO369ruMWidjGp6s6n4', 'Map: kernel', style="filled", fillcolor="red")
dot.node('AMcFTURFNKQ6GzjFxQav', 'Call: sqrt', style="filled", fillcolor="green")
dot.edge('ktZZFrrMGHOnrk7m0ACZ', 'ALUCx51bmHH1HgAIf7o1')
dot.edge('ktZZFrrMGHOnrk7m0ACZ', 'kme7ittgG00LuD5h7qbO')
dot.edge('ktZZFrrMGHOnrk7m0ACZ', 'xNEIrJyFnm3mYPdmSuWI')
dot.edge('ktZZFrrMGHOnrk7m0ACZ', 'XkAiVGFt3I9y6ZIEKibL')
dot.edge('ktZZFrrMGHOnrk7m0ACZ', '0tb0X4Gjm86PusO2J2SB')
dot.edge('ktZZFrrMGHOnrk7m0ACZ', 'GhiRA0fJBKNdPFkHN2D1')
dot.edge('ktZZFrrMGHOnrk7m0ACZ', '9IO369ruMWidjGp6s6n4')
dot.edge('9IO369ruMWidjGp6s6n4', 'AMcFTURFNKQ6GzjFxQav')



print(dot.source)
dot.render('nn_Call_Tree.gv', view=True)