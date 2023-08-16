<#assign gh = glex.getGlobalVar("pfHelper")>
${signature("returnStatement")}
return <#if returnStatement.isPresentReturnExpression()> ${gh.printExpression(returnStatement.getReturnExpression())} </#if>;