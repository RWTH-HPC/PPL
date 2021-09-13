from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('SIYg41unlDlKGZHyGamM', 'Main')
dot.node('cLJIqJJqDxzMKZVa7tGF', 'Expression')
dot.node('1T6RK93a3GEcM7n2ofgR', 'Call: init_List', style="filled", fillcolor="green")
dot.node('XM2uDT0KjJfXoPyCcRFD', 'Expression')
dot.node('Ktt8N9UBiJ15a13AdVa3', 'Call: init_List', style="filled", fillcolor="green")
dot.node('B4mAslVtU6EXALPWeiIq', 'Expression')
dot.node('E4S8XfCYLIUS1yzgXWJ9', 'Call: init_List', style="filled", fillcolor="green")
dot.node('ZPGOnOINKxehxFUN3YuC', 'Expression')
dot.node('7eQAf9uy05pHBEUtO9Mq', 'Call: init_List', style="filled", fillcolor="green")
dot.node('WSZ2M4zQ8WOeKAzzASvZ', 'Expression')
dot.node('mx1bA2YzXpiHFtzHvtsD', 'Call: init_List', style="filled", fillcolor="green")
dot.node('pCbykxP6jH6RAZ99h8XI', 'Expression')
dot.node('AIgPa9ZRA5z3oDIlF3fB', 'Call: init_List', style="filled", fillcolor="green")
dot.node('LkAzp74bQUZIjXhvKY1A', 'Simple Expression')
dot.node('WgIJbqk7824HsRBhTZ6P', 'Map: kernel', style="filled", fillcolor="red")
dot.node('hnDMarD3N8HyYubvBntQ', 'Expression')
dot.node('MfFhFG25ujOWUc1pgPr4', 'Expression')
dot.node('Bo55EhD67IfGnQKWEpoy', 'Call: sqrt', style="filled", fillcolor="green")
dot.node('oH7dtzA222uDg8kpPG7m', 'Simple Expression')
dot.node('l9ld28DUTcoQ7ZVtGA2y', 'Return')
dot.node('DFCUtkGOZYIZKxZbI7Nd', 'Expression')
dot.node('F95pAs58DzRlRUrrzJOD', 'For-Loop')
dot.node('FFNTz2HIU8RAN0yzZ38x', 'Expression')
dot.node('DkzdqUEEta2eGPHlzVyE', 'Expression')
dot.node('gqpfWYU2yARZgbUPclyE', 'Expression')
dot.node('DYv9jk6Y6D6RDSJTgb9f', 'Simple Expression')
dot.node('cME9HYafQPnjVTQMTWH7', 'For-Loop')
dot.node('Ir7YdlGzlmJzTATNCecp', 'Expression')
dot.node('oGUdpRYvLBKWqUiEgCkL', 'Expression')
dot.node('0bjpon7R1HzgYf29VvTC', 'Expression')
dot.node('dCIQZbqNNKN0JdXBZZQR', 'Branch')
dot.node('6ORvryg9FWDNl3Y22tEd', 'Case')
dot.node('PSU2iWebZzOcZXS6O7yx', 'Expression')
dot.node('8DIGZZT6prDpgDRHVkPi', 'Simple Expression')
dot.node('GLV64449hiZpKNYetZm8', 'Branch')
dot.node('UDIMRVyAibsZMhj81KdG', 'Case')
dot.node('UFRTGYRcbRL6in2LVtdE', 'Expression')
dot.node('ln75QXEsshVskgYYo3aC', 'Simple Expression')
dot.node('9IaE298AyKjtKd2VP8Qu', 'Simple Expression')
dot.node('BFEER1IKv3qQxaMqkHJ2', 'Return')
dot.node('7JS7Vi1Er9f0NXaQGoDK', 'Expression')
dot.edge('SIYg41unlDlKGZHyGamM', 'cLJIqJJqDxzMKZVa7tGF')
dot.edge('SIYg41unlDlKGZHyGamM', 'XM2uDT0KjJfXoPyCcRFD')
dot.edge('SIYg41unlDlKGZHyGamM', 'B4mAslVtU6EXALPWeiIq')
dot.edge('SIYg41unlDlKGZHyGamM', 'ZPGOnOINKxehxFUN3YuC')
dot.edge('SIYg41unlDlKGZHyGamM', 'WSZ2M4zQ8WOeKAzzASvZ')
dot.edge('SIYg41unlDlKGZHyGamM', 'pCbykxP6jH6RAZ99h8XI')
dot.edge('SIYg41unlDlKGZHyGamM', 'LkAzp74bQUZIjXhvKY1A')
dot.edge('SIYg41unlDlKGZHyGamM', 'WgIJbqk7824HsRBhTZ6P')
dot.edge('SIYg41unlDlKGZHyGamM', 'F95pAs58DzRlRUrrzJOD')
dot.edge('SIYg41unlDlKGZHyGamM', '9IaE298AyKjtKd2VP8Qu')
dot.edge('SIYg41unlDlKGZHyGamM', 'BFEER1IKv3qQxaMqkHJ2')
dot.edge('cLJIqJJqDxzMKZVa7tGF', '1T6RK93a3GEcM7n2ofgR')
dot.edge('XM2uDT0KjJfXoPyCcRFD', 'Ktt8N9UBiJ15a13AdVa3')
dot.edge('B4mAslVtU6EXALPWeiIq', 'E4S8XfCYLIUS1yzgXWJ9')
dot.edge('ZPGOnOINKxehxFUN3YuC', '7eQAf9uy05pHBEUtO9Mq')
dot.edge('WSZ2M4zQ8WOeKAzzASvZ', 'mx1bA2YzXpiHFtzHvtsD')
dot.edge('pCbykxP6jH6RAZ99h8XI', 'AIgPa9ZRA5z3oDIlF3fB')
dot.edge('WgIJbqk7824HsRBhTZ6P', 'hnDMarD3N8HyYubvBntQ')
dot.edge('WgIJbqk7824HsRBhTZ6P', 'MfFhFG25ujOWUc1pgPr4')
dot.edge('MfFhFG25ujOWUc1pgPr4', 'Bo55EhD67IfGnQKWEpoy')
dot.edge('Bo55EhD67IfGnQKWEpoy', 'oH7dtzA222uDg8kpPG7m')
dot.edge('Bo55EhD67IfGnQKWEpoy', 'l9ld28DUTcoQ7ZVtGA2y')
dot.edge('l9ld28DUTcoQ7ZVtGA2y', 'DFCUtkGOZYIZKxZbI7Nd')
dot.edge('F95pAs58DzRlRUrrzJOD', 'FFNTz2HIU8RAN0yzZ38x')
dot.edge('F95pAs58DzRlRUrrzJOD', 'DkzdqUEEta2eGPHlzVyE')
dot.edge('F95pAs58DzRlRUrrzJOD', 'gqpfWYU2yARZgbUPclyE')
dot.edge('F95pAs58DzRlRUrrzJOD', 'DYv9jk6Y6D6RDSJTgb9f')
dot.edge('F95pAs58DzRlRUrrzJOD', 'cME9HYafQPnjVTQMTWH7')
dot.edge('F95pAs58DzRlRUrrzJOD', 'GLV64449hiZpKNYetZm8')
dot.edge('cME9HYafQPnjVTQMTWH7', 'Ir7YdlGzlmJzTATNCecp')
dot.edge('cME9HYafQPnjVTQMTWH7', 'oGUdpRYvLBKWqUiEgCkL')
dot.edge('cME9HYafQPnjVTQMTWH7', '0bjpon7R1HzgYf29VvTC')
dot.edge('cME9HYafQPnjVTQMTWH7', 'dCIQZbqNNKN0JdXBZZQR')
dot.edge('dCIQZbqNNKN0JdXBZZQR', '6ORvryg9FWDNl3Y22tEd')
dot.edge('6ORvryg9FWDNl3Y22tEd', 'PSU2iWebZzOcZXS6O7yx')
dot.edge('6ORvryg9FWDNl3Y22tEd', '8DIGZZT6prDpgDRHVkPi')
dot.edge('GLV64449hiZpKNYetZm8', 'UDIMRVyAibsZMhj81KdG')
dot.edge('UDIMRVyAibsZMhj81KdG', 'UFRTGYRcbRL6in2LVtdE')
dot.edge('UDIMRVyAibsZMhj81KdG', 'ln75QXEsshVskgYYo3aC')
dot.edge('BFEER1IKv3qQxaMqkHJ2', '7JS7Vi1Er9f0NXaQGoDK')



print(dot.source)
dot.render('nn_Complete_Tree.gv', view=True)