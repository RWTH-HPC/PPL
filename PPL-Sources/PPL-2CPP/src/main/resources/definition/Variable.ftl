<#compress>
${signature("variable", "prefix")}
<#assign gh = glex.getGlobalVar("pfHelper")> <#-- GeneratorHelper -->
${gh.printType(variable.getType())} ${prefix}${variable.getName()}<#if variable.isPresentExpression() && gh.isNotContainerType(variable) && gh.isNotClassType(variable)> =</#if><#if variable.isPresentExpression()>  ${gh.printExpression(variable.getExpression())}</#if>;
</#compress>