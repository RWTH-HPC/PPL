from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('VdJ9MkfE5iRUEDy3Qs5c', 'Main')
dot.node('qedpzKd6iSzsbevAvxm3', 'Call: init_List', style="filled", fillcolor="green")
dot.node('NxKiggsNBWcLL6zGvunl', 'Call: init_List', style="filled", fillcolor="green")
dot.node('pZnIY5E8Gvy41ltMvOKO', 'Call: init_List', style="filled", fillcolor="green")
dot.node('JXPu9V4HlBBlcnAL1NFR', 'Call: init_List', style="filled", fillcolor="green")
dot.node('fUo0KcD6LubRL9cFJdsx', 'Map: initialize_vars', style="filled", fillcolor="red")
dot.node('9z6ZxB8ACzSbNgyZ3eK7', 'Call: init_List', style="filled", fillcolor="green")
dot.node('cnAGFtCzLRFo6YZENFQd', 'Call: init_List', style="filled", fillcolor="green")
dot.node('lxYQIqdbfu2akgmVNX4E', 'Call: init_List', style="filled", fillcolor="green")
dot.node('cBA4DINOteYPpAI0YeWh', 'Map: copy', style="filled", fillcolor="red")
dot.node('WpbPC0r2aG7JvRXkEDyU', 'Map: compute_step_factor', style="filled", fillcolor="red")
dot.node('1YkLQzYPWtJautlVtYhA', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('Z61DzPzU7pUsU9qV8JZX', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('D4cUKgyLuOjLlD5T1MBt', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('WAEpWyYtciQC0ugUPEdA', 'Call: compute_speed_sqd', style="filled", fillcolor="green")
dot.node('Vrs9cjpUQzk5TVjMVyBI', 'Call: compute_pressure', style="filled", fillcolor="green")
dot.node('pqiEGuRqtd1jhk4LFWD6', 'Call: compute_speed_of_sound', style="filled", fillcolor="green")
dot.node('2z6BY4coqzQaYmH1N1Xe', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('p3UCsZWgtj8BGAlE0hxe', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('UCn0APkqcoSy0GuYPGJy', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('JLsYdjbEoDUx1dvLfGRi', 'Map: compute_flux', style="filled", fillcolor="red")
dot.node('PLRibhAmIrvpTTTgXNV8', 'Call: init_List', style="filled", fillcolor="green")
dot.node('dg1yeyQ4sR8UjMPAI9WO', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('bpJG3KikTh4or6zpELjY', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('vu9PYzohutuvUb8pnVnn', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('FgscEztc5Ep98knySQd5', 'Call: compute_speed_sqd', style="filled", fillcolor="green")
dot.node('mdz0qIgpTcYITSIZCIkf', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('rtEKnobuiruIXopFEa20', 'Call: compute_pressure', style="filled", fillcolor="green")
dot.node('rlIREjN4CD5Ufdd4JTF7', 'Call: compute_speed_of_sound', style="filled", fillcolor="green")
dot.node('jZNiTyaqrjx6X3SOgM3k', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('u0ZexN9RpmBmNJ1acqLP', 'Call: init_List', style="filled", fillcolor="green")
dot.node('JW8mWiBDmsm2pUPdRLkE', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('96KjqRMK51o8CBXQdQ89', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('7SrPyZi7Kd9nSOm6cVTL', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('IFZXSHhCV2D8LtD68JL2', 'Call: compute_velocity', style="filled", fillcolor="green")
dot.node('bR5lYrVa4paItE4THBzV', 'Call: compute_speed_sqd', style="filled", fillcolor="green")
dot.node('JQVvnmLGR4aOvrOgxx7z', 'Call: compute_pressure', style="filled", fillcolor="green")
dot.node('MQuBuQPyAhtlm4EUWshr', 'Call: compute_speed_of_sound', style="filled", fillcolor="green")
dot.node('lkg6sjjJ6ncH3R3N6HpK', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('B0epl0KuzKMCYa3OynsP', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('pSng2FUnYry2OifeX5V0', 'Map: time_step', style="filled", fillcolor="red")
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'qedpzKd6iSzsbevAvxm3')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'NxKiggsNBWcLL6zGvunl')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'pZnIY5E8Gvy41ltMvOKO')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'JXPu9V4HlBBlcnAL1NFR')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'fUo0KcD6LubRL9cFJdsx')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', '9z6ZxB8ACzSbNgyZ3eK7')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'cnAGFtCzLRFo6YZENFQd')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'lxYQIqdbfu2akgmVNX4E')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'cBA4DINOteYPpAI0YeWh')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'WpbPC0r2aG7JvRXkEDyU')
dot.edge('WpbPC0r2aG7JvRXkEDyU', '1YkLQzYPWtJautlVtYhA')
dot.edge('WpbPC0r2aG7JvRXkEDyU', 'Z61DzPzU7pUsU9qV8JZX')
dot.edge('WpbPC0r2aG7JvRXkEDyU', 'D4cUKgyLuOjLlD5T1MBt')
dot.edge('WpbPC0r2aG7JvRXkEDyU', 'WAEpWyYtciQC0ugUPEdA')
dot.edge('WpbPC0r2aG7JvRXkEDyU', 'Vrs9cjpUQzk5TVjMVyBI')
dot.edge('WpbPC0r2aG7JvRXkEDyU', 'pqiEGuRqtd1jhk4LFWD6')
dot.edge('pqiEGuRqtd1jhk4LFWD6', '2z6BY4coqzQaYmH1N1Xe')
dot.edge('WpbPC0r2aG7JvRXkEDyU', 'p3UCsZWgtj8BGAlE0hxe')
dot.edge('WpbPC0r2aG7JvRXkEDyU', 'UCn0APkqcoSy0GuYPGJy')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'JLsYdjbEoDUx1dvLfGRi')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'PLRibhAmIrvpTTTgXNV8')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'dg1yeyQ4sR8UjMPAI9WO')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'bpJG3KikTh4or6zpELjY')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'vu9PYzohutuvUb8pnVnn')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'FgscEztc5Ep98knySQd5')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'mdz0qIgpTcYITSIZCIkf')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'rtEKnobuiruIXopFEa20')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'rlIREjN4CD5Ufdd4JTF7')
dot.edge('rlIREjN4CD5Ufdd4JTF7', 'jZNiTyaqrjx6X3SOgM3k')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'u0ZexN9RpmBmNJ1acqLP')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'JW8mWiBDmsm2pUPdRLkE')
dot.edge('JLsYdjbEoDUx1dvLfGRi', '96KjqRMK51o8CBXQdQ89')
dot.edge('JLsYdjbEoDUx1dvLfGRi', '7SrPyZi7Kd9nSOm6cVTL')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'IFZXSHhCV2D8LtD68JL2')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'bR5lYrVa4paItE4THBzV')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'JQVvnmLGR4aOvrOgxx7z')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'MQuBuQPyAhtlm4EUWshr')
dot.edge('MQuBuQPyAhtlm4EUWshr', 'lkg6sjjJ6ncH3R3N6HpK')
dot.edge('JLsYdjbEoDUx1dvLfGRi', 'B0epl0KuzKMCYa3OynsP')
dot.edge('VdJ9MkfE5iRUEDy3Qs5c', 'pSng2FUnYry2OifeX5V0')



print(dot.source)
dot.render('cfd_Call_Tree.gv', view=True)