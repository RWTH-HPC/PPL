from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('lxtoIAWy2QbDdaT6jtbA', 'Main')
dot.node('pTiQvibdG2MaobkkddvE', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('UmFTLopWi0rDoPdLY5Ok', 'Map: normalize', style="filled", fillcolor="orangered")
dot.node('htNYEcheNfH3I3ktamWn', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('2GTNXFi6rGEEHyMqt5nq', 'Map: inner_normalization', style="filled", fillcolor="orangered")
dot.node('EVUNAnzJupf4v5mqnDWT', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('uSovIcmNSY5vq0lrTnn2', 'Map: extract', style="filled", fillcolor="orangered")
dot.node('WKiCUn88KHchpZEYnzvj', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('lzTorlhjLLs7GZ0NTpyY', 'Reduction: single_feature', style="filled", fillcolor="orangered")
dot.node('edcQU7300xqEdnYFfxAM', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('V2zmV6oYO0UXTOKVYqN2', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('kMMk4eechXsXJeYvihWQ', 'Map: classify', style="filled", fillcolor="orangered")
dot.node('ozyOpqgC5TfRYVbEuQC3', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('20DAotmzln5jDUiVkLZB', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('lxtoIAWy2QbDdaT6jtbA', 'pTiQvibdG2MaobkkddvE')
dot.edge('lxtoIAWy2QbDdaT6jtbA', 'UmFTLopWi0rDoPdLY5Ok')
dot.edge('UmFTLopWi0rDoPdLY5Ok', 'htNYEcheNfH3I3ktamWn')
dot.edge('UmFTLopWi0rDoPdLY5Ok', '2GTNXFi6rGEEHyMqt5nq')
dot.edge('2GTNXFi6rGEEHyMqt5nq', 'EVUNAnzJupf4v5mqnDWT')
dot.edge('lxtoIAWy2QbDdaT6jtbA', 'uSovIcmNSY5vq0lrTnn2')
dot.edge('uSovIcmNSY5vq0lrTnn2', 'WKiCUn88KHchpZEYnzvj')
dot.edge('uSovIcmNSY5vq0lrTnn2', 'lzTorlhjLLs7GZ0NTpyY')
dot.edge('lzTorlhjLLs7GZ0NTpyY', 'edcQU7300xqEdnYFfxAM')
dot.edge('uSovIcmNSY5vq0lrTnn2', 'V2zmV6oYO0UXTOKVYqN2')
dot.edge('lxtoIAWy2QbDdaT6jtbA', 'kMMk4eechXsXJeYvihWQ')
dot.edge('kMMk4eechXsXJeYvihWQ', 'ozyOpqgC5TfRYVbEuQC3')
dot.edge('lxtoIAWy2QbDdaT6jtbA', '20DAotmzln5jDUiVkLZB')



print(dot.source)
dot.render('batch_Pattern_Nesting_Tree.gv', view=True)