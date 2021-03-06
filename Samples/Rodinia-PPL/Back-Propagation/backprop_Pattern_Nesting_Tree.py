from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('eeQvNvXd5tXqK1rlifMf', 'Main')
dot.node('tOHU047RQ7mzgCc8K8Km', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('SD7MZLqXkzZM78XXAYsa', 'Map: forward_pass', style="filled", fillcolor="orangered")
dot.node('NrSsBLKviGR9TCj84gNq', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('CaPsbcXvO0izoEOrRt8Y', 'Reduction: weighted_sum', style="filled", fillcolor="orangered")
dot.node('2cZpkFIpYxhSIdGShY4v', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('JYfcq5gFfnShfvK317fb', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('LGU0BFAeDNK3ixtzaFGd', 'Map: forward_pass', style="filled", fillcolor="orangered")
dot.node('aqWJM0TX3zguhh4vmZUK', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('RnDxiArd8o3TdSBxROF5', 'Reduction: weighted_sum', style="filled", fillcolor="orangered")
dot.node('RioKlg2Voej1d6bSnJZS', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('ngCRljtqv9mXs0HOSGFp', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('C81UNvj5z8uNTguTl4WA', 'Map: output_error', style="filled", fillcolor="orangered")
dot.node('R4ie0B3Dv0c95Kcpd4s0', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('MBxdbIWGxNGFZeQHrtRM', 'Stencil: traverse', style="filled", fillcolor="orangered")
dot.node('pBCIQ2H0tcNlNKzggSrm', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('GkshO7OlEYVrflUNVZ24', 'Map: hidden_error', style="filled", fillcolor="orangered")
dot.node('2yG2lGX1QNtrcQlCop5S', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('zUyCyGTdMMCK9tuFxJkK', 'Reduction: weighted_sum', style="filled", fillcolor="orangered")
dot.node('Dl8RE6VcnDvfNBZvcJ08', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('Jf8OuRziov2x7yjjleW8', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('IZXEl26iePErRGk134py', 'Stencil: traverse', style="filled", fillcolor="orangered")
dot.node('RdS3qaVToq8tCBLECidZ', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('VaqD2KZB8hhNRceJuBxT', 'Stencil: move', style="filled", fillcolor="orangered")
dot.node('Q9lMSOeAD8jDYN0AlN2V', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('PYQKGHgUuP6V5tM1dUEe', 'Stencil: update_weights', style="filled", fillcolor="orangered")
dot.node('C4oMaMXYuf8CtGeeFWvu', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('tR4hGGFKaWiLIo8c5GVA', 'Stencil: update_weights', style="filled", fillcolor="orangered")
dot.node('MsiJG0ee1ryOkfFt963p', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('STRL1IQJbf473BUhhUO9', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('eeQvNvXd5tXqK1rlifMf', 'tOHU047RQ7mzgCc8K8Km')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'SD7MZLqXkzZM78XXAYsa')
dot.edge('SD7MZLqXkzZM78XXAYsa', 'NrSsBLKviGR9TCj84gNq')
dot.edge('SD7MZLqXkzZM78XXAYsa', 'CaPsbcXvO0izoEOrRt8Y')
dot.edge('CaPsbcXvO0izoEOrRt8Y', '2cZpkFIpYxhSIdGShY4v')
dot.edge('SD7MZLqXkzZM78XXAYsa', 'JYfcq5gFfnShfvK317fb')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'LGU0BFAeDNK3ixtzaFGd')
dot.edge('LGU0BFAeDNK3ixtzaFGd', 'aqWJM0TX3zguhh4vmZUK')
dot.edge('LGU0BFAeDNK3ixtzaFGd', 'RnDxiArd8o3TdSBxROF5')
dot.edge('RnDxiArd8o3TdSBxROF5', 'RioKlg2Voej1d6bSnJZS')
dot.edge('LGU0BFAeDNK3ixtzaFGd', 'ngCRljtqv9mXs0HOSGFp')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'C81UNvj5z8uNTguTl4WA')
dot.edge('C81UNvj5z8uNTguTl4WA', 'R4ie0B3Dv0c95Kcpd4s0')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'MBxdbIWGxNGFZeQHrtRM')
dot.edge('MBxdbIWGxNGFZeQHrtRM', 'pBCIQ2H0tcNlNKzggSrm')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'GkshO7OlEYVrflUNVZ24')
dot.edge('GkshO7OlEYVrflUNVZ24', '2yG2lGX1QNtrcQlCop5S')
dot.edge('GkshO7OlEYVrflUNVZ24', 'zUyCyGTdMMCK9tuFxJkK')
dot.edge('zUyCyGTdMMCK9tuFxJkK', 'Dl8RE6VcnDvfNBZvcJ08')
dot.edge('GkshO7OlEYVrflUNVZ24', 'Jf8OuRziov2x7yjjleW8')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'IZXEl26iePErRGk134py')
dot.edge('IZXEl26iePErRGk134py', 'RdS3qaVToq8tCBLECidZ')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'VaqD2KZB8hhNRceJuBxT')
dot.edge('VaqD2KZB8hhNRceJuBxT', 'Q9lMSOeAD8jDYN0AlN2V')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'PYQKGHgUuP6V5tM1dUEe')
dot.edge('PYQKGHgUuP6V5tM1dUEe', 'C4oMaMXYuf8CtGeeFWvu')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'tR4hGGFKaWiLIo8c5GVA')
dot.edge('tR4hGGFKaWiLIo8c5GVA', 'MsiJG0ee1ryOkfFt963p')
dot.edge('eeQvNvXd5tXqK1rlifMf', 'STRL1IQJbf473BUhhUO9')



print(dot.source)
dot.render('backprop_Pattern_Nesting_Tree.gv', view=True)