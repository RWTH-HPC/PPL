from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('jA0ND0kALrEgpPI1UHfk', 'Main')
dot.node('SxMZkWcgzzpR3GoBOaeX', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('1y8pAQkTtWGtzneC9eKp', 'Map: initialize_vars', style="filled", fillcolor="orangered")
dot.node('JjFQhFP5fK6ZaRfiPA9D', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('pg65KTXbm9DbVIuGvXTQ', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('iDr8gYgKS3Gp5RDCnJVC', 'For Loop', style="filled", fillcolor="cyan")
dot.node('ti5cy9VWe0lhdTi0pzG4', 'Map: copy', style="filled", fillcolor="orangered")
dot.node('VSLSq4oeDipfqHOdLxzG', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('12SNFnWBKBUPergOUlhQ', 'Map: compute_step_factor', style="filled", fillcolor="orangered")
dot.node('mjSqSdUK9kyvGaOWsvJZ', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('M2cz2f7r2fUB8UY8ykS9', 'For Loop', style="filled", fillcolor="cyan")
dot.node('NeDSLDo6Gko2TDWYtyvM', 'Map: compute_flux', style="filled", fillcolor="orangered")
dot.node('R2WxjDNVY60OlMxtih5e', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('QOlBe2Ve7ejdUbzvs0tb', 'Map: time_step', style="filled", fillcolor="orangered")
dot.node('plRBCrxaRtfSsroHI8dF', 'Serial', style="filled", fillcolor="lawngreen")
dot.node('pWWuimhm1eZA1baI9UkR', 'Serial', style="filled", fillcolor="lawngreen")
dot.edge('jA0ND0kALrEgpPI1UHfk', 'SxMZkWcgzzpR3GoBOaeX')
dot.edge('jA0ND0kALrEgpPI1UHfk', '1y8pAQkTtWGtzneC9eKp')
dot.edge('1y8pAQkTtWGtzneC9eKp', 'JjFQhFP5fK6ZaRfiPA9D')
dot.edge('jA0ND0kALrEgpPI1UHfk', 'pg65KTXbm9DbVIuGvXTQ')
dot.edge('jA0ND0kALrEgpPI1UHfk', 'iDr8gYgKS3Gp5RDCnJVC')
dot.edge('iDr8gYgKS3Gp5RDCnJVC', 'ti5cy9VWe0lhdTi0pzG4')
dot.edge('ti5cy9VWe0lhdTi0pzG4', 'VSLSq4oeDipfqHOdLxzG')
dot.edge('iDr8gYgKS3Gp5RDCnJVC', '12SNFnWBKBUPergOUlhQ')
dot.edge('12SNFnWBKBUPergOUlhQ', 'mjSqSdUK9kyvGaOWsvJZ')
dot.edge('iDr8gYgKS3Gp5RDCnJVC', 'M2cz2f7r2fUB8UY8ykS9')
dot.edge('M2cz2f7r2fUB8UY8ykS9', 'NeDSLDo6Gko2TDWYtyvM')
dot.edge('NeDSLDo6Gko2TDWYtyvM', 'R2WxjDNVY60OlMxtih5e')
dot.edge('M2cz2f7r2fUB8UY8ykS9', 'QOlBe2Ve7ejdUbzvs0tb')
dot.edge('QOlBe2Ve7ejdUbzvs0tb', 'plRBCrxaRtfSsroHI8dF')
dot.edge('jA0ND0kALrEgpPI1UHfk', 'pWWuimhm1eZA1baI9UkR')



print(dot.source)
dot.render('cfd_Pattern_Nesting_Tree.gv', view=True)