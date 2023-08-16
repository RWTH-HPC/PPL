<#assign gh = glex.getGlobalVar("pfHelper")>
${signature("ifStatement")}
if(${gh.printExpression(ifStatement.getCondition())})
{${includeArgs("statement/BlockStatement.ftl",ifStatement.getThenStatement())}}
<#if ifStatement.getElseStatementOpt().isPresent()>
    else <#if ifStatement.getElseStatement().getBlockStatementOpt().isPresent()>{${includeArgs("statement/BlockStatement.ftl",ifStatement.getElseStatement().getBlockStatement())} }<#else> ${includeArgs("statement/IfStatement.ftl",ifStatement.getElseStatement().getIfStatement())}  </#if>
</#if>