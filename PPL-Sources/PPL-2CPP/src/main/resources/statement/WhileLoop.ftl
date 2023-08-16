<#assign gh = glex.getGlobalVar("pfHelper")>
${signature("loop")}
while ( ${gh.printExpression(loop.getCondition())} ) {${includeArgs("statement/BlockStatement.ftl",loop.getBlockStatement())}}
