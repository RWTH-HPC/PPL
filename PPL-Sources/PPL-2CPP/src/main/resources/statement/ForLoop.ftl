<#compress>
<#assign gh = glex.getGlobalVar("pfHelper")> <#-- GeneratorHelper -->
${signature("loop")}
<#assign loopControl=loop.getForControl()>
<#if gh.isCommonForControl(loopControl)>
<@compress single_line=true> for (
    <#assign vars=loopControl.getForInit()>
    <#if vars.getExpressionOpt().isPresent()>
        ${gh.printExpression(vars.getExpression())}
    </#if>
    <#if vars.getVariableOpt().isPresent()>
        ${includeArgs("definition/Variable.ftl", vars.getVariable(), "")}
    </#if>;
    <#assign controlString="">
    ${gh.printExpression(loopControl.getCondition())};
    <#assign exp=loopControl.getExpression()>
    ${gh.printExpression(exp)}
    </@compress>)
</#if>
<#if gh.isForEachControl(loopControl)>
    for(auto ${loopControl.getName()} : ${gh.printExpression(loopControl.getExpression())})
</#if>
</#compress>
{${includeArgs("statement/BlockStatement.ftl", loop.getBlockStatement())}}