from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS
<#assign gh=glex.getGlobalVar("pfHelper")> <#-- GeneratorHelper -->


${gh.generateTree()}


print(dot.source)
dot.render('${gh.getName()}.gv', view=True)